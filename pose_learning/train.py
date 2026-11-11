"""Train the pose head; fixed final checkpoint, Train-only diagnostics."""
import argparse
import json
from pathlib import Path
import math
import random
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True,help='Completed Train feature cache')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=20000)
    parser.add_argument('--microbatch',type=int,default=1)
    parser.add_argument('--effective-batch',type=int,default=4)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--warmup',type=int,default=500)
    parser.add_argument('--seed',type=int,default=6198)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--precision',choices=('bf16','float32'),default='bf16')
    parser.add_argument('--checkpoint-every',type=int,default=1000)
    parser.add_argument('--diagnostic-every',type=int,default=500)
    parser.add_argument('--allow-smoke-cache',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if min(args.updates,args.microbatch,args.effective_batch,args.checkpoint_every,args.diagnostic_every)<1 or args.effective_batch%args.microbatch:
        parser.error('Positive counts required; effective batch must be divisible by microbatch')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or args.warmup<0:
        parser.error('Invalid learning rate or warmup')
    print('Loading Train-only features and pose-head dependencies',flush=True)
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from .data import CachedPoseDataset,fit_normalization,write_json,digest,TARGET_FRAME
    from .model import PoseHead,pose_loss
    from .geometry import quaternion_to_matrix,rotation_error_deg
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if args.precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        parser.error('BF16 requires a supported CUDA GPU; use --precision float32 for CPU smoke')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(args.seed)
    generator=torch.Generator().manual_seed(args.seed+1)
    norm=fit_normalization(args.cache)
    dataset=CachedPoseDataset(args.cache,'train',norm)
    if dataset.metadata['smoke_limited'] and not args.allow_smoke_cache:
        parser.error('Limited cache is for smoke only; explicit --allow-smoke-cache required')
    signature=digest(args.cache/'metadata.json')
    print(f'Validated {len(dataset)} Train observations; initializing pose head',flush=True)
    model=PoseHead().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    print(f'Optimizer ready; requested updates={args.updates}, effective batch={args.effective_batch}',flush=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    start_update=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
        if checkpoint['cache_metadata_sha256']!=signature or checkpoint['target_frame']!=TARGET_FRAME:
            raise ValueError('Resume source/pose-frame mismatch')
        for key in ('updates','microbatch','effective_batch','learning_rate','weight_decay','warmup','seed','precision'):
            if checkpoint['configuration'][key]!=config[key]:raise ValueError('Resume configuration changed: '+key)
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        start_update=int(checkpoint['completed_updates'])
        torch.set_rng_state(checkpoint['torch_rng'])
        generator.set_state(checkpoint['sample_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        if args.output.resolve()!=args.resume.resolve().parent.parent:
            parser.error('Resume must use the original output directory')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'checkpoints').mkdir()
    write_json(args.output/'configuration.json',dict(config,normalization=norm,model=model.configuration,
        cache_metadata_sha256=signature,cache_metadata=dataset.metadata,trainable_parameters=sum(p.numel() for p in model.parameters()),
        target_frame=TARGET_FRAME,checkpoint_selection='predetermined final optimizer update, no Test access',
        data_sampling='uniform observed conditioning pairs with replacement; four examples per source trajectory',
        supervision_caveat='Source phase/closure/flag heuristic, not force-certified physical grasp success'))
    def pack(indices):
        batch=default_collate([dataset[int(i)] for i in indices])
        return {name:value.to(device) for name,value in batch.items()}
    def save(update):
        destination=args.output/'checkpoints'/f'step{update:05d}.pt'
        temporary=destination.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=update,
            configuration=config,model_configuration=model.configuration,normalization=norm,target_frame=TARGET_FRAME,
            cache_metadata_sha256=signature,cache_metadata=dataset.metadata,torch_rng=torch.get_rng_state(),
            sample_rng=generator.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),temporary)
        temporary.replace(destination)
        # Keep the last scheduled checkpoint and every 5k milestone.
        for previous in (args.output/'checkpoints').glob('step*.pt'):
            number=int(previous.stem[4:])
            if number!=update and number%5000!=0:previous.unlink()
        return str(destination)
    started=time.monotonic();latest=str(args.resume) if args.resume else None;completed=start_update
    if start_update>=args.updates:
        write_json(args.output/'status.json',dict(state='complete',completed_updates=start_update,
            requested_updates=args.updates,latest_checkpoint=latest,test_accessed=False))
        return
    write_json(args.output/'status.json',dict(state='running',completed_updates=start_update,requested_updates=args.updates,test_accessed=False))
    try:
        with (args.output/'train.jsonl').open('a' if args.resume else 'x') as log:
            for update in range(start_update+1,args.updates+1):
                model.train();optimizer.zero_grad(set_to_none=True)
                losses=[]
                factor=args.effective_batch//args.microbatch
                for _ in range(factor):
                    indices=torch.randint(len(dataset),(args.microbatch,),generator=generator)
                    batch=pack(indices)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
                        output=model(**batch);loss,terms=pose_loss(output,batch)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pose loss')
                    (loss/factor).backward();losses.append(float(loss.detach()))
                gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                warmup=min(1.,update/max(args.warmup,1))
                progress=max(0.,(update-args.warmup)/max(args.updates-args.warmup,1))
                learning_rate=args.learning_rate*warmup*(.1+.9*.5*(1+math.cos(math.pi*min(progress,1))))
                for group in optimizer.param_groups:group['lr']=learning_rate
                optimizer.step()
                completed=update
                record=dict(update=update,loss=float(np.mean(losses)),gradient_norm=gradient,learning_rate=learning_rate,
                    elapsed_seconds=time.monotonic()-started)
                if update%args.diagnostic_every==0 or update==args.updates:
                    model.eval();diagnostics=[]
                    with torch.inference_mode():
                        for start in range(0,min(64,len(dataset)),8):
                            batch=pack(range(start,min(start+8,64,len(dataset))))
                            output=model(**batch)
                            diagnostics.append(dict(position=(output['position']-batch['target'][:,:3]).norm(dim=-1).cpu().numpy(),
                                rotation=rotation_error_deg(output['rotation'],quaternion_to_matrix(batch['target'][:,3:])).cpu().numpy(),
                                arm=(output['arm_logits'].argmax(-1)==batch['arm']).float().cpu().numpy()))
                    record['train_only_diagnostic']=dict(observations=sum(len(x['position']) for x in diagnostics),
                        position_mean_m=float(np.concatenate([x['position'] for x in diagnostics]).mean()),
                        rotation_mean_deg=float(np.concatenate([x['rotation'] for x in diagnostics]).mean()),
                        arm_accuracy=float(np.concatenate([x['arm'] for x in diagnostics]).mean()),
                        used_for_checkpoint_selection=False)
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
                if update==1 or update%100==0 or update==args.updates:print(json.dumps(record),flush=True)
                if update%args.checkpoint_every==0 or update==args.updates:latest=save(update)
                if update%100==0 or update==args.updates:
                    write_json(args.output/'status.json',dict(state='complete' if update==args.updates else 'running',
                        completed_updates=update,requested_updates=args.updates,latest_checkpoint=latest,test_accessed=False))
    except BaseException as error:
        write_json(args.output/'status.json',dict(state='failed',completed_updates=completed,
            requested_updates=args.updates,error=repr(error),test_accessed=False))
        raise


if __name__=='__main__':main()
