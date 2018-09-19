import sys
from snor import SnorIter

if len(sys.argv) != 3:
    print("Usage: find_missing_hyp_ids.py <input-file> <filtered-ids>")
    sys.exit(1)

input_file = sys.argv[1]
filtered_ids_file = sys.argv[2]

def main():

    with open(input_file, 'r', encoding='utf-8') as input_fh, open(filtered_ids_file, 'r', encoding='utf-8') as filtered_ids_fh:
        filtered_ids = set()
        for line in filtered_ids_fh:
            filtered_ids.add(line.strip().capitalize())

        for utt, uttid in SnorIter(input_fh):
            if uttid not in filtered_ids:
                print("{0} ({1})".format(utt, uttid))

if __name__ == "__main__":
    main()
