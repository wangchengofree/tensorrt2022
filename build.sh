python3 modelTrans.py

trtexec --workspace=3000 --onnx=./encoder_fp16.onnx --optShapes=speech:4x64x80,speech_lengths:4 --maxShapes=speech:16x256x80,speech_lengths:16 --minShapes=speech:1x16x80,speech_lengths:1 --shapes=speech:1x16x80,speech_lengths:1 --saveEngine=./encoder.plan

trtexec --workspace=30000 --onnx=/workspace/decoder.onnx --optShapes=encoder_out:4x64x256,encoder_out_lens:4,typs_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 --maxShapes==encoder_out:16x256x256,encoder_out_lens:16,typs_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 --minShapes==encoder_out:1x16x256,encoder_out_lens:1,typs_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 --saveEngine=./decoder.plan
