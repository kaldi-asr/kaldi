#!/usr/bin/env bash

# Intel MKL is now freely available even for commercial use. This script
# attempts to install the MKL package automatically from Intel's repository.
#
# For manual repository setup instructions, see:
#   https://software.intel.com/articles/installing-intel-free-libs-and-python-yum-repo
#   https://software.intel.com/articles/installing-intel-free-libs-and-python-apt-repo
#
# For other package managers, or non-Linux platforms, see:
#   https://software.intel.com/mkl/choose-download

set -o pipefail

default_package=intel-mkl-64bit-2020.0-088

yum_repo='https://yum.repos.intel.com/mkl/setup/intel-mkl.repo'
apt_repo='https://apt.repos.intel.com/mkl'
intel_key_url='https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB'

Usage () {
  cat >&2 <<EOF
Usage: $0 [-s] [<MKL-package>]

Checks if MKL is present on the system, and/or attempts to install it.

If <MKL-package> is not provided, ${default_package} will be installed.

Intel packages are installed under the /opt/intel directory. You should be root
to install MKL into this directory; run this script using the sudo command.

Options:
  -s  - Skip check for MKL being already present.
  -p <suse|redhat|debian|fedora> -- Force type of package management. Use only
                                    if automatic detection fails, as instructed.
  -h  - Show this message.

Environment:
  CC   The C compiler to use for MKL check. If not set, uses 'cc'.
EOF
  exit 2
}

Fatal () { echo "$0: $@"; exit 1; }

Have () { type -t "$1" >/dev/null; }

# Option values.
skip_cc=
distro=

while getopts ":hksp:" opt; do
  case ${opt} in
    h) Usage ;;
    s) skip_cc=yes ;;
    p) case $OPTARG in
         suse|redhat|debian|fedora) distro=$OPTARG ;;
         *) Fatal "invalid value -p '${OPTARG}'. " \
                  "Allowed: 'suse', 'redhat', 'debian' or 'fedora'."
       esac ;;
    \?) echo >&2 "$0: invalid option -${OPTARG}."; Usage ;;
  esac
done
shift $((OPTIND-1))

orig_arg_package=${1-''}
package=${1:-$default_package}

# Check that we are actually on Linux, otherwise give a helpful reference.
[[ $(uname) == Linux ]] || Fatal "\
This script can be used on Linux only, and your system is $(uname).

Installer packages for Mac and Windows are available for download from Intel:
https://software.intel.com/mkl/choose-download"

# Test if MKL is already installed on the system.
if [[ ! $skip_cc ]]; then
  : ${CC:=cc}
  Have "$CC" || Fatal "\
C compiler $CC not found.

You can skip the check for MKL presence by invoking this script with the '-s'
option to this script, but you will need a functional compiler anyway, so we
recommend that you install it first."

  mkl_version=$($CC -E -I /opt/intel/mkl/include - <<< \
                      '#include <mkl_version.h>
           __INTEL_MKL__.__INTEL_MKL_MINOR__.__INTEL_MKL_UPDATE__' 2>/dev/null |
                  tail -n 1 ) || mkl_version=
  mkl_version=${mkl_version// /}

  [[ $mkl_version ]] && Fatal "\
MKL version $mkl_version is already installed.

You can skip the check for MKL presence by invoking this script with the '-s'
option and proceed with automated installation, but we highly discourage
this. This script will register Intel repositories with your system, and it
seems that they have been already registered, or MKL has been installed some
other way.

You should use your package manager to check which MKL package is already
installed. Note that Intel packages register the latest installed version of
the library as the default. If your installed version is older than
$package, it makes sense to upgrade."
fi

# Try to determine which package manager the distro uses, unless overridden.
if [[ ! $distro ]]; then
  dist_vars=$(cat /etc/os-release 2>/dev/null)
  eval "$dist_vars"
  for rune in $CPE_NAME $ID $ID_LIKE; do
    case "$rune" in
      cpe:/o:fedoraproject:fedora:2[01]) distro=redhat; break;;  # Use yum.
      rhel|centos) distro=redhat; break;;
      redhat|suse|fedora|debian) distro=$rune; break;;
    esac
  done

  # Certain old distributions do not have /etc/os-release. We are unlikely to
  # encounter these in the wild, but just in case.
  # NOTE: Do not try to guess Fedora specifically here! Fedora 20 and below
  #       detect as redhat, and this is good, because they use yum by default.
  [[ ! $distro && -f /etc/redhat-release ]] && distro=redhat
  [[ ! $distro && -f /etc/SuSE-release ]]   && distro=suse
  [[ ! $distro && -f /etc/debian_release ]] && distro=debian
  [[ ! $distro && -f /etc/arch-release ]] && distro=arch

  [[ ! $distro ]] && Fatal "\
Unable to determine package management style.

Invoke this script with the option '-p <style>', where <style> can be:
  redhat -- RedHat-like, uses yum and rpm for package management.
  fedora -- Fedora 22+, also RedHat-like, but uses dnf instead of yum.
  suse   -- SUSE-like, uses zypper and rpm.
  debian -- Debian-like, uses apt and dpkg.
  arch   -- Archlinux, uses pacman.

We do not currently support other package management systems. Check the Intel's
documentation at https://software.intel.com/mkl/choose-download for other
install options."

  echo >&2 "$0: Your system is using ${distro}-style package management."
fi

# Check for root.
if [[ "$(id -u)" -ne 0 ]]; then
  echo >&2 "$0: You must be root to install MKL.

Restart this script using the 'sudo' command, as:

  sudo $0 -sp $distro $package

We recommend adding the '-sp $distro' options to skip the MKL and distro
detection, since this has already been done. This minimizes the number of
programs invoked with the root privileges to keep your system safe from
unexpected or erroneous changes. Also, if you are setting the CC environment
variable, sudo might not allow it to propagate to the command that it invokes."

  if [ -t 0 ]; then
    echo; read -ep "Run the above sudo command now? [Y/n]:"
    case $REPLY in
      ''|[Yy]*) set -x; exec sudo "$0" -sp "$distro" "$package"
    esac
  fi
  exit 0
fi

# The install variants, each in a function to simplify error reporting.
# Each one invokes a subshell with a 'set -x' to to show system-modifying
# commands it runs. The subshells simply limit the scope of this diagnostics
# and avoid creating noise (if we were using 'set +x', it would be printed).
Install_redhat () {
  # yum-utils contains yum-config-manager, in case the user does not have it.
  ( set -x
    rpm --import $intel_key_url
    yum -y install yum-utils &&
    yum-config-manager --add-repo "$yum_repo" &&
    yum -y install "$package" )
}

Install_fedora () {
  ( set -x
    rpm --import $intel_key_url
    dnf -y install 'dnf-command(config-manager)' &&
    dnf config-manager --add-repo "$yum_repo" &&
    dnf -y install "$package" )
}

Install_suse () {
  # zypper bug until libzypp-17.6.4: '--gpg-auto-import-keys' is ignored.
  # See https://github.com/openSUSE/zypper/issues/144#issuecomment-418685933
  # We must disable gpg checks with '--no-gpg-checks'. I won't bend backwards
  # as far as check the installed .so version...
  ( set -x
    rpm --import $intel_key_url
    zypper addrepo "$yum_repo" &&
    zypper --gpg-auto-import-keys --no-gpg-checks \
           --non-interactive install "$package" )
}

Install_debian () {
  local keyring='/usr/share/keyrings/intel-sw-products.gpg' \
        sources_d='/etc/apt/sources.list.d' \
        trusted_d='/etc/apt/trusted.gpg.d' \
        apt_maj= apt_min= apt_ver=

  # apt before 1.2 does not understand the signed-by option, and always
  # look for the keyring in their trusted.gpg.d directory. This is not
  # considered a good security practice any more. If apt is old, add a link
  # to the keyring file and remind the user to delete it when apt is upgraded.
  IFS=' .' builtin read _ apt_maj apt_min _ < <(apt-get --version)
  apt_ver=$(builtin printf '%03d%03d' $apt_maj $apt_min)

  # Get alternative location of /etc/apt/sources.list.d, if so configured.
  eval $(apt-config shell sources_d Dir::Etc::sourceparts/f \
                          trusted_d Dir::Etc::trustedparts/f)

  # apt is much more involved to configure than other package managers, as fas
  # as third-party security keys go.
  ( set -x;
    apt-get update &&
    apt-get install -y wget apt-transport-https ca-certificates gnupg &&
    wget -qO- $intel_key_url | apt-key --keyring $keyring add - &&
    echo "deb [signed-by=${keyring}] $apt_repo all main" \
         > "$sources_d/intel-mkl.list" ) || return 1

  if [[ $apt_ver < '001002' ]]; then
    ( set -x; ln -s "$keyring" "${trusted_d}/" ) || return 1
  fi

  ( set +x
    apt-get update &&
    apt-get install -y "$package" ) || return 1

  # Print the message after the large install, so the user may notice. I hope...
  if [[ $apt_ver < '001002' ]]; then
    echo >&2 "$0: Your apt-get version is earlier than 1.2.

This version does not understand individual repositories signing keys, and
trusts all keys in $trusted_d. We have created a link
$trusted_d/$(basename $keyring) pointing to the file
$keyring. If/when you upgrade your system to
a higher version of apt, removing this link will help make it more secure.

This is not considered a severe security issue, but separating keyrings is the
current recommended security practice."
  fi
}

Install_arch () {
  ( set -x
    echo y | pacman -Syu intel-mkl && # In pacman we don't specify the version
    pacman -Q --info intel-mkl | grep -v None
  )
}

# Register MKL .so libraries with the ld.so.
ConfigLdSo() {
  [ -d /etc/ld.so.conf.d ] || return 0
  type -t ldconfig >/dev/null || return 0
  echo >&2 "$0: Configuring ld runtime bindings"
  ( set -x;
    echo >/etc/ld.so.conf.d/intel-mkl.conf "\
/opt/intel/lib/intel64
/opt/intel/mkl/lib/intel64"
    ldconfig )
}

# Invoke installation.
if Install_${distro} && ConfigLdSo; then
  echo >&2 "$0: MKL package $package was successfully installed"
else
  Fatal "MKL package $package installation FAILED.

Please open an issue with us at https://github.com/kaldi-asr/kaldi/ if you
believe this is a bug."
fi
