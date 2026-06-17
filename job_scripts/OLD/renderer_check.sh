#!/bin/bash
#PBS -N renderer_check
#PBS -l select=1:ncpus=2:ngpus=1:mem=4gb
#PBS -l walltime=00:05:00

nvidia-smi

echo "EGL libs:"
ldconfig -p | grep -i egl

echo "Renderer:"
glxinfo -B 2>/dev/null | grep -i renderer
