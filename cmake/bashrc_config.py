import argparse
import os
import shutil

PATH_BASHRC = os.path.join(os.environ['HOME'], '.bashrc')
PATH_BASHRC_BACKUP = os.path.join(os.environ['HOME'], '.bashrc~')
CFG_START_STRING = "# >>> kaldi initialize >>>"
CFG_END_STRING = "# <<< kaldi initialize <<<"


def get_kaldi_setting():
    bashrc_core = []
    kaldi_setting = []
    with open(PATH_BASHRC, 'r') as infile:
        file_content = infile.readlines()
        is_kaldi_setting = False

        for line in file_content:
            line = line.strip()  # type: str
            if line.startswith(CFG_START_STRING):
                kaldi_setting.append(line)
                is_kaldi_setting = True
            elif line.startswith(CFG_END_STRING):
                kaldi_setting.append(line)
                is_kaldi_setting = False
            elif is_kaldi_setting:
                kaldi_setting.append(line)
            else:
                bashrc_core.append(line)

    return bashrc_core, kaldi_setting


def install_kaldi_setting(kaldi_dir):
    bashrc_core, _ = get_kaldi_setting()
    bashrc_core.append(CFG_START_STRING)
    bashrc_core.append("""export KALDI_DIR=%s
echo "Kaldi prefix path: %s"  """ % (kaldi_dir, kaldi_dir))
    bashrc_core.append(CFG_END_STRING)
    shutil.copy(PATH_BASHRC, PATH_BASHRC_BACKUP)

    final_bashrc_content = '\n'.join(bashrc_core)
    with open(PATH_BASHRC, 'w') as outfile:
        outfile.write(final_bashrc_content)


def uninstall_kaldi_setting():
    bashrc_core, _ = get_kaldi_setting()

    shutil.copy(PATH_BASHRC, PATH_BASHRC_BACKUP)

    final_bashrc_content = '\n'.join(bashrc_core)
    with open(PATH_BASHRC, 'w') as outfile:
        outfile.write(final_bashrc_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Installation of Kaldi environment variable to ~/.bashrc")
    parser.add_argument("kaldi_dir", nargs=1, type=str,
                        help="Kaldi Installation Prefix, named by `KALDI_DIR`")
    config = parser.parse_args()

    kaldi_dir = config.kaldi_dir[0]

    uninstall_kaldi_setting()
    install_kaldi_setting(kaldi_dir)
