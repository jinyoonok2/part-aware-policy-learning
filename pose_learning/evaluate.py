"""Offline pose prediction errors, distinct from physical door-opening success."""
import argparse
import json
from pathlib import Path


def summarize(array):
    import numpy as np
    # Arm correctness is boolean; NumPy percentile interpolates using numeric
    # subtraction, so convert all metrics to an explicit floating-point array.
    array=np.asarray(array,dtype=np.float64)
    return dict(mean=float(np.mean(array)),median=float(np.median(array)),p90=float(np.percentile(array,90)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--role',choices=('test','train'),default='test')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('Batch size must be positive')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import CachedPoseDataset,TARGET_FRAME,write_json,digest
    from .geometry import quaternion_to_matrix,rotation_error_deg
    from .model import PoseHead
    torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint['target_frame']!=TARGET_FRAME or checkpoint['cache_metadata']['role']!='train':
        raise ValueError('Checkpoint was not trained with the declared Train pose frame')
    dataset=CachedPoseDataset(args.cache,args.role,checkpoint['normalization'])
    if dataset.metadata['encoder_sha256']!=checkpoint['cache_metadata']['encoder_sha256']:
        raise ValueError('Evaluation encoder differs from training')
    if args.role=='test' and dataset.metadata['smoke_limited']:
        raise ValueError('Final Test cannot use an incomplete smoke cache')
    args.output.mkdir(parents=True,exist_ok=False)
    model=PoseHead(**checkpoint['model_configuration']).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    values={name:[] for name in ('position_error_m','orientation_error_deg','arm_correct','finger_span_error_m')}
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0):
            batch={name:value.to(args.device) for name,value in batch.items()}
            output=model(**batch)
            values['position_error_m'].extend((output['position']-batch['target'][:,:3]).norm(dim=-1).cpu().tolist())
            values['orientation_error_deg'].extend(rotation_error_deg(output['rotation'],quaternion_to_matrix(batch['target'][:,3:])).cpu().tolist())
            values['arm_correct'].extend((output['arm_logits'].argmax(-1)==batch['arm']).cpu().tolist())
            values['finger_span_error_m'].extend((output['gripper_width']-batch['gripper_width_m']).abs().cpu().tolist())
    examples=json.loads((args.cache/'examples.json').read_text())
    if len(examples)!=len(dataset):raise ValueError('Example identity alignment mismatch')
    from collections import defaultdict
    grouped=defaultdict(list)
    with (args.output/'predictions.jsonl').open('x') as stream:
        for i,example in enumerate(examples):
            record=dict(house=example['house'],h5=example['h5'],trajectory=example['trajectory'],frame=example['frame'],
                **{name:value[i] for name,value in values.items()})
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            grouped[(example['h5'],example['trajectory'])].append(i)
    report=dict(role=args.role,scope='offline pose errors; does not measure physical grasp or door opening',
        observations=len(dataset),trajectories=len(grouped),completed_training_updates=checkpoint['completed_updates'],
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=digest(args.checkpoint),cache_metadata_sha256=digest(args.cache/'metadata.json'),
        target_frame=TARGET_FRAME,normalization_fitted_on='Train only',checkpoint_selection='predetermined final update; this evaluation does not select a checkpoint',
        per_observation={name:summarize(value) for name,value in values.items()},
        per_trajectory_mean={name:summarize([np.mean(np.asarray(value)[indices]) for indices in grouped.values()]) for name,value in values.items()},
        orientation_error_includes_wrong_arm_predictions=True,
        label_confidence='Source heuristic initial-grasp labels; no force-certified outcome claim')
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
