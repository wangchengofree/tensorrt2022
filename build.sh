python3 modelTrans.py

trtexec --workspace=3000 --onnx=./encoder_fp16.onnx --optShapes=speech:4x64x80,speech_lengths:4 --maxShapes=speech:16x256x80,speech_lengths:16 --minShapes=speech:1x16x80,speech_lengths:1 --shapes=speech:1x16x80,speech_lengths:1 --saveEngine=./encoder.plan
