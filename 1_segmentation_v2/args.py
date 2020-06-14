import argparse
def get_parser():

  parser = argparse.ArgumentParser(description='Slurm-based DSC_driver')
  # mode
  parser.add_argument("-m", "--mode", type=str, choices=['train','predict'], required=True,
                      help="Enter a run mode for the script")
  parser.add_argument("-i", "--input-path", type=str, dest='input_path', required=True,
                      help="Path to training or prediction data folder")
  parser.add_argument("-o", "--output-path", type=str, dest='output_path', required=True,
                      help="Path to save segmented images")
  parser.add_argument("-w", "--weights-path", type=str, dest='weights_path', required=True,
                      help="Path to checkpointed weights")
  parser.add_argument('-ps', dest='patch_size', default = 128, type=int,
                      help = "patch_size")
  parser.add_argument('-dt', dest='data_type', default = '*.png', type=str,
                      help = "image data type")
  parser.add_argument("-q", "--quiet", dest='verbose', action="store_false",
                    help="Disables verbose printing")
  return parser.parse_args()

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()