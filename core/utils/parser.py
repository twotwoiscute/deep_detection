import argparse 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",
                        type=str,
                        required=True, 
                        help="The config file for detection."
    )
    args = parser.parse_args()
    return args
