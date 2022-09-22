import util
import scenario

def main():
    args = util.get_args()
    util.print_args(args)
    method = getattr(scenario, args.scenario)
    try:
        method(args)
    except KeyboardInterrupt:
        pass
        
if __name__ == '__main__':
    main()
