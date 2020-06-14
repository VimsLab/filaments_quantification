#!/usr/bin/env python
from __future__ import print_function
import os, sys, argparse, shlex, subprocess
from args import get_parser

def convertTuple(tup): 
    str =  "".join(tup) 
    return str

def setup_slurm_script(args):
  input_path = args.input_path
  output_path = args.output_path


  debug_print = print if args.verbose else lambda *args: None

  # Path to this script
  self_path = os.path.realpath(__file__)
  self_dir = os.path.dirname(self_path)
  matlab_code_path = os.path.join(self_dir, '/matlabCodes/') 
  sbatch_flags = ["-N 1",
                "-c 1",
                "--mem=16000",

                ]

  debug_print ("============================")
  debug_print ("Slurm sbatch flags:")
  debug_print ("\n".join(sbatch_flags))

  debug_print (matlab_code_path)
#                   "-dt" + " " + convertTuple(data_type))

  # segment_command = ["export PATH=/opt/MATLAB/R2018b/bin", "matlab", "-nodisplay", 
  #                  "-nojvm",  "-r", 
  #                  "cd " + matlab_code_path + "" + ";main\("+ input_path +"," +  output_path + "\)", 
  #                 ]

  segment_command = ["export PATH=/opt/MATLAB/R2018b/bin", "matlab", "-nodisplay", 
                   "-nojvm", 
                   "cd " + matlab_code_path + "",  "-r", "main", 
                  ]

  debug_print(segment_command)         
  segment_command_str = " ".join(segment_command)

  debug_print ("============================")
  debug_print ("Segment command:")
  debug_print (segment_command_str)
  segment_command_wrapped = '--wrap=" ' + segment_command_str + '"'
  # Construct sbatch command
  sbatch_command = ["sbatch"] + sbatch_flags + [segment_command_wrapped]
  sbatch_command_str = " ".join(sbatch_command)
  debug_print ("============================")
  debug_print ("Slurm sbatch command:")
  debug_print (sbatch_command_str)
  
  print ("============================")
  final_sbatch_command = shlex.split(sbatch_command_str)
  p = subprocess.Popen(final_sbatch_command)
  p.wait()
  print ("Success!")
  print ("============================")
  sys.stdout.flush()
  
if __name__ == "__main__":

    args = get_parser()
    setup_slurm_script(args)