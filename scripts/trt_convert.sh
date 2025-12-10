#!/usr/bin/env bash
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 input.onnx output.trt"
  exit 1
fi
onnx_file=$1
trt_file=$2
trtexec --onnx="${onnx_file}" --saveEngine="${trt_file}" --fp16 || exit 1
echo "Saved TRT model to ${trt_file}"
