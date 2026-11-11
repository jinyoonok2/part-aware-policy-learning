"""Cache observed frozen DINO features and pose labels for exactly one data role."""
import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--role',choices=('train','test'),required=True)
    parser.add_argument('--encoder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--limit',type=int,help='Train-only smoke limit, explicitly recorded; never a final Test subset')
    parser.add_argument('--verify-h5',action='store_true',help='Recompute all raw source hashes before feature extraction')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and (args.limit < 1 or args.role != 'train')):
        parser.error('Use a positive batch size and a positive Train-only smoke limit')
    started_full=time.monotonic()
    import h5py
    import numpy as np
    import torch
    from decord import VideoReader
    from .data import load_examples, observed_points, state_vector, decode, digest, write_json, TARGET_FRAME, POINT_RULE, CAMERAS
    from .vision import FrozenVision
    torch.set_num_threads(2)
    all_examples = load_examples(args.index,args.role)
    examples = all_examples[:args.limit] if args.limit else all_examples
    args.output.mkdir(parents=True,exist_ok=False)
    encoder_file = args.encoder/'model.safetensors'
    metadata = dict(complete=False,role=args.role,index=str(args.index.resolve()),index_sha256=digest(args.index),
        target_frame=TARGET_FRAME,point_rule=POINT_RULE,encoder_path=str(args.encoder.resolve()),
        encoder_sha256=digest(encoder_file),observations=len(examples),full_role_observations=len(all_examples),
        smoke_limited=args.limit is not None,camera_order=CAMERAS,visual_transform='whole RGB image stretched to 224x224, ImageNet normalization',
        raw_hashes_reverified=args.verify_h5,inputs='current RGB, current qpos excluding global base, current visible part points, push/pull task',
        supervision='future early initial-grasp TCP; source metadata heuristic, not force-certified firm grasp')
    write_json(args.output/'metadata.json',metadata)
    write_json(args.output/'examples.json',examples)
    if args.verify_h5:
        for path in sorted({e['h5'] for e in examples}):
            expected={e['source_h5_sha256'] for e in examples if e['h5']==path}
            if expected != {digest(path)}:
                raise ValueError('Raw HDF5 source hash mismatch: '+path)
    print(f'Loading frozen local DINOv2 encoder on {args.device}',flush=True)
    encoder = FrozenVision(args.encoder,args.device)
    print('Frozen encoder loaded; decoding current observations',flush=True)
    visual = np.lib.format.open_memmap(args.output/'features.npy', mode='w+', dtype=np.float16, shape=(len(examples),3,2,768))
    states, points, targets, arms, widths, tasks = [],[],[],[],[],[]
    started=time.monotonic()
    # Bound open file/video resources to the current trajectory. The index is
    # trajectory-grouped; do not retain thousands of FFmpeg decoder contexts.
    current=None; h5=None; readers={}; source=None
    try:
        for start in range(0,len(examples),args.batch_size):
            images=[]; batch_points=[]
            for example in examples[start:start+args.batch_size]:
                key=(example['h5'],example['trajectory'])
                if key != current:
                    if h5 is not None: h5.close()
                    readers.clear()
                    h5=h5py.File(example['h5'],'r'); source=h5[example['trajectory']]
                    if len(source['obs/agent/qpos']) != example['source_observations']:
                        raise ValueError('Source observation count changed')
                    readers={camera:VideoReader(example['videos'][camera],num_threads=1) for camera in CAMERAS}
                    if any(len(r)!=example['source_observations'] for r in readers.values()):
                        raise ValueError('Video/state observation alignment changed')
                    current=key
                frame=example['frame']
                observed=observed_points(source,frame)
                images.append({camera:readers[camera][frame].asnumpy() for camera in CAMERAS})
                batch_points.append(observed)
                states.append(state_vector(decode(source['obs/agent/qpos'][frame])))
                points.append(observed);targets.append(example['target']);arms.append(example['arm'])
                widths.append(example['gripper_width_m']);tasks.append(example['is_pull'])
            features=encoder(images,np.stack(batch_points)).cpu().numpy().astype(np.float16)
            if not np.isfinite(features).all(): raise ValueError('Nonfinite visual features')
            visual[start:start+len(features)]=features
            if start % 100 == 0:
                print(f'{start+len(features)}/{len(examples)} observed frames encoded',flush=True)
    finally:
        if h5 is not None:h5.close()
    visual.flush();del visual
    np.savez_compressed(args.output/'arrays.npz',state=np.asarray(states,np.float32),points=np.asarray(points,np.float32),
        target=np.asarray(targets,np.float32),arm=np.asarray(arms,np.int64),gripper_width_m=np.asarray(widths,np.float32),
        is_pull=np.asarray(tasks,np.float32))
    metadata.update(complete=True,encoding_elapsed_seconds=time.monotonic()-started,
        wall_elapsed_seconds=time.monotonic()-started_full,
        files_sha256={name:digest(args.output/name) for name in ('features.npy','arrays.npz','examples.json')})
    write_json(args.output/'metadata.json',metadata)
    print(f'Complete: {len(examples)} {args.role} observations cached in {args.output}',flush=True)


if __name__=='__main__':main()
