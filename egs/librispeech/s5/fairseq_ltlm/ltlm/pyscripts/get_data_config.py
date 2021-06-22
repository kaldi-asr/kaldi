import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs="+", required=True,
                        help="Train.  format 'lats,[suff:default empty],[epoch:default 0] lats2 lat3,suff,epoch'")
    parser.add_argument('--valid', nargs="+", required=True, help="Valid. format 'lats,ref lat2,ref2'")
    parser.add_argument('--test', nargs='*', default=None, help="Test. format like valid")
    parser.add_argument("--out", type=str, required=True, help="Path to result data_config.json file")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = {"train": [], }
    for l_s_e in args.train:
        l_s_e_sep = l_s_e.split(',')
        l = l_s_e_sep[0]
        s = l_s_e_sep[1] if len(l_s_e_sep) > 1 else ''
        e = l_s_e_sep[2] if len(l_s_e_sep) > 2 else ''
        e = int(e if e else 0)
        data['train'].append({"epoch": e, "lats": l, "utt_suff": s})

    data["valid"] = []
    for l_r in args.valid:
        l_r_sep = l_r.split(',')
        assert len(l_r_sep) == 2, RuntimeError("Wrong valid format. Must be lat,ref lat2,ref ...")
        l, r = l_r_sep
        data['valid'].append({'lats': l, 'ref': r})
    if args.test:
        data['test'] = []
        for l_r in args.test:
            l_r_sep = l_r.split(',')
            assert len(l_r_sep) == 2, RuntimeError("Wrong test format. Must be lat,ref lat2,ref ...")
            l, r = l_r_sep
            data['test'].append({'lats': l, 'ref': r})
    with open(args.out, 'w') as out:
        json.dump(data, out, indent=4)


if __name__ == "__main__":
    main()
