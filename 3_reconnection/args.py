import argparse
def get_parser():

  parser = argparse.ArgumentParser(description='matlab reconnection script')
  # mode
  parser.add_argument("-i", "--input-path", type=str, dest='input_path', required=True,
                      help="Path to training or prediction data folder")
  parser.add_argument("-o", "--output-path", type=str, dest='output_path', required=True,
                      help="Path to save segmented images")
  parser.add_argument("-q", "--quiet", dest='verbose', action="store_false",
                    help="Disables verbose printing")
  return parser.parse_args()

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()