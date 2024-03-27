#python train_gendata.py --noautoanchor --workers 32 --batch-size 32 --epochs 80 --plate-style 'realrandom'
#python train_gendata.py --noautoanchor --workers 32 --batch-size 32 --epochs 80 --plate-style 'realrandom' --resume
python train_gendata.py --noautoanchor --workers 32 --batch-size 32 --epochs 80 --plate-style 'real' --resume --data plate_cocowocar.yaml

