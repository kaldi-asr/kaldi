#!/bin/bash
# Copyright 2015  Johns Hopkins University (author: Vijayaditya Peddinti)
# Apache 2.0
# This script downloads the impulse responses from http://www.openairlib.net/
# It downloads only impulse responses and not sinesweep responses
#==============================================

download=true
sampling_rate=8k
output_bit=16
DBname=OPENAIR
file_splitter=  #script to generate job scripts given the command file

. cmd.sh
. path.sh
. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <rir-home> <output-dir> <log-dir>"
  echo "e.g.:"
  echo " $0  --download true db/RIR_databases/ data/impulses_noises exp/make_reverb/log"
  exit 1;
fi

RIR_home=$1
output_dir=$2
log_dir=$3

if [ "$download" = true ]; then
  mkdir -p $RIR_home
  RIR_home_abs=`readlink -e $RIR_home`
  #HamiltonMuseum 
  #http://www.openairlib.net/auralizationdb/content/hamilton-mausoleum 
  echo "">$log_dir/${DBname}_download_commands.sh
  dir=$RIR_home_abs/open_air/hamilton_mausoleum/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/hamilton-mausoleum/stereo/hm2_000_ortf_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/hamilton-mausoleum/b-format/hm2_000_bformat_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/hamilton-mausoleum/surround-5-1/hm_williams.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Terry's Factory Warehouse 
  dir=$RIR_home_abs/open_air/terry_factory_warehouse/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-factory-warehouse/mono/terrys_warehouse_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-factory-warehouse/stereo/terrys_warehouse_stereo.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-factory-warehouse/b-format/terrys_warehouse_b_format.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-factory-warehouse/surround-5-1/terrys_warehouse_5-0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Tyndall Bruce Monument
  dir=$RIR_home_abs/open_air/tyndall_bruce_monument/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/tyndall-bruce-monument/mono/tyndall_bruce_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/tyndall-bruce-monument/stereo/tyndall_bruce_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/tyndall-bruce-monument/b-format/tyndall_bruce_b_format.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/tyndall-bruce-monument/surround-5-1/tyndall_bruce_5-0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #R1 Nuclear Reactor Hall
  dir=$RIR_home_abs/open_air/r1_nuclear_reactor_hall/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/mono/r1_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/mono/r1_omni_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/stereo/r1_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/stereo/r1_ortf-48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/b-format/r1_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/b-format/r1_bformat-48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/surround-5-1/r1_williams.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/r1-nuclear-reactor-hall/surround-5-1/r1_williams-48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Koli National Park - Winter
  dir=$RIR_home_abs/open_air/koli_national_park/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site1_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site1_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site2_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site2_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site3_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site3_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site4_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/mono/koli_snow_site4_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site1_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site1_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site2_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site2_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site3_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site3_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site4_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/koli-national-park-winter/b-format/koli_snow_site4_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Elveden Hall
  dir=$RIR_home_abs/open_air/elvenden_hall/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mrogalsky/elveden-hall-suffolk-england/stereo/1a_marble_hall.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mrogalsky/elveden-hall-suffolk-england/stereo/3a_hats_cloaks_the_lord.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mrogalsky/elveden-hall-suffolk-england/stereo/4a_hats_cloaks_visitors.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mrogalsky/elveden-hall-suffolk-england/stereo/18a_smoking_room.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Troller's Grill
  dir=$RIR_home_abs/open_air/trollers_grill/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/mono/dales_site1_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/mono/dales_site1_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/mono/dales_site2_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/mono/dales_site2_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/mono/dales_site3_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/mono/dales_site3_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/b-format/dales_site1_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/b-format/dales_site1_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/b-format/dales_site2_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/b-format/dales_site2_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/b-format/dales_site3_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/trollers-gill/b-format/dales_site3_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Domestic living room
  dir=$RIR_home_abs/open_air/domestic_living_room/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/ligidium/domestic-living-room/stereo/living_room_1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/ligidium/domestic-living-room/stereo/living_room_2.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/ligidium/domestic-living-room/stereo/living_room_in_bedroom_1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Centro Cultural de Cascais
  #not used as this is not an impulse response
  # but a sinusoidal sweep
  #dir=$RIR_home_abs/open_air/centro_cultural_de_cascais/
  #echo "wget http://www.openairlib.net/sites/default/files/auralization/data/asilva/centro-cultural-de-cascais/b-format/df.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  #echo "wget http://www.openairlib.net/sites/default/files/auralization/data/asilva/centro-cultural-de-cascais/b-format/dt.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  #echo "wget http://www.openairlib.net/sites/default/files/auralization/data/asilva/centro-cultural-de-cascais/b-format/ef.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  #echo "wget http://www.openairlib.net/sites/default/files/auralization/data/asilva/centro-cultural-de-cascais/b-format/et.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Terry's Typing Room
  dir=$RIR_home_abs/open_air/terry_typing_room/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-typing-room/mono/terrys_typing_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-typing-room/stereo/terrys_typing_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-typing-room/b-format/terrys_typing_b_format.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/terrys-typing-room/surround-5-1/terrys_typing_5-0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #The Lady Chapel, St Albans Cathedral
  dir=$RIR_home_abs/open_air/lady_chapel_stalbans_cathedral/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/mono/stalbans_a_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/mono/stalbans_b_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/mono/stalbans_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/stereo/stalbans_a_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/stereo/stalbans_b_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/stereo/stalbans_a_binaural.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/stereo/stalbans_b_binaural.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/b-format/stalbans_a_wxyz.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/b-format/stalbans_b_wxyz.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/surround-5-1/stalbans_a_williams.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/lady-chapel-st-albans-cathedral/surround-5-1/stalbans_b_williams.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Innocent Railway Tunnel
  dir=$RIR_home_abs/open_air/innocent_railway_tunnel/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/middle_tunnel_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/middle_tunnel_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_a_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_b_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_c_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_d_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_e_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_f_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_a_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_b_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_c_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_d_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_e_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/mono/tunnel_entrance_f_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/middle_tunnel_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/middle_tunnel_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_a_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_b_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_c_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_d_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_e_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_f_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_a_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_b_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_c_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_d_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_e_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/innocent-railway-tunnel/b-format/tunnel_entrance_f_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Maes Howe
  dir=$RIR_home_abs/open_air/maes_howe/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/maes-howe/stereo/mh3_000_ortf_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/maes-howe/b-format/mh3_000_bformat_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Koli National Park - Summer
  dir=$RIR_home_abs/open_air/koli_national_park_summer/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site1_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site1_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site2_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site2_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site3_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site3_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site4_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/mono/koli_summer_site4_4way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site1_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site1_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site2_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site2_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site3_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site3_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site4_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/koli-national-park-summer/b-format/koli_summer_site4_4way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #St. Margaret's Church - National Centre for Early Music
  dir=$RIR_home_abs/open_air/stmargarets_church/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r1_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r2_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r3_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r4_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r5_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r6_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r7_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r8_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r9_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r10_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r11_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r12_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r13_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r14_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r15_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r16_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r17_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r18_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r19_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r20_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r21_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r22_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r23_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r24_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r25_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r26_1st_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r1_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r2_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r3_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r4_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r5_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r6_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r7_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r8_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r9_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r10_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r11_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r12_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r13_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r14_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r15_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r16_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r17_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r18_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r19_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r20_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r21_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r22_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r23_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r24_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r25_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r26_2nd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r1_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r2_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r3_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r4_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r5_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r6_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r7_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r8_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r9_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r10_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r11_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r12_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r13_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r14_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r15_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r16_3rd_configuration_0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r17_3rd_configuration_0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r18_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r19_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r20_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r21_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r22_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r23_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r24_3rd_configuration.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r25_3rd_configuration_0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/afoteinou/st-margarets-church-national-centre-early-music/b-format/r26_3rd_configuration_0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Alcuin College, University of York
  dir=$RIR_home_abs/open_air/alcuin_college/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s1r1_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s1r1front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s1r1_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s1r2_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s1r2front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s1r2_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s1r3_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s1r3front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s1r3_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s2r1_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s2r1front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s2r1_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s2r2_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s2r2front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s2r2_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s2r3_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s2r3front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s2r3_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s3r1_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s3r1front_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s3r1_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s4r1_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s4r1_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/alcuin_s4r2_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/fstevens/alcuin-college-university-york/b-format/s4r2_spist_bform.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Spokane Woman's Club 
  dir=$RIR_home_abs/open_air/spokane_womans_club/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/top-jimmy/spokane-womans-club/stereo/spokane_womans_club_ir.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Stairway, University of York
  dir=$RIR_home_abs/open_air/stairway_university_of_york/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/simon/stairway-university-york/stereo/stairwell_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Underground Car Park
  dir=$RIR_home_abs/open_air/underground_car_park/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/chrisunderdown/underground-car-park/mono/carpark_balloon_ir_mono_24bit_44100.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/chrisunderdown/underground-car-park/stereo/carpark_balloon_ir_stereo_24bit_44100.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Falkland Palace Bottle Dungeon
  dir=$RIR_home_abs/open_air/falkland_palace_bottle_dungeon/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/falkland-palace-bottle-dungeon/b-format/bottledungeon1_sf_edited.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Railroad tunnel - Purnode's Tunnel
  dir=$RIR_home_abs/open_air/railroad_purnodes_tunnel/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/valvalion/railroad-tunnel-purnodes-tunnel/stereo/ir_purnode_tunnel_balloon_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #St Andrew's Church
  dir=$RIR_home_abs/open_air/standrews_church/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-andrews-church/stereo/lyd3_000_ortf_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-andrews-church/b-format/lyd3_000_bformat_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Sports Centre, University of York
  dir=$RIR_home_abs/open_air/sports_centre_university_of_york/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/sports-centre-university-york/mono/sportscentre_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/sports-centre-university-york/mono/sportscentre_cardioid.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/sports-centre-university-york/stereo/sportscentre_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/sports-centre-university-york/b-format/sportscentre_wxyz.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/sports-centre-university-york/surround-5-1/sportscentre_williams.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Basement 
  dir=$RIR_home_abs/open_air/basement/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s2.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s3.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s4.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s5.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s6.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s7.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s8.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/mganger/basement/stereo/s9.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Falkland Palace Royal Tennis Court
  dir=$RIR_home_abs/open_air/falkland_palace_royal_tennis_court/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/falkland-palace-royal-tennis-court/mono/falkland_tennis_court_omni.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/falkland-palace-royal-tennis-court/stereo/falkland_tennis_court_ortf.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/falkland-palace-royal-tennis-court/b-format/falkland_tennis_court_b_format.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/falkland-palace-royal-tennis-court/surround-5-1/falkland_tennis_court_5-0.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #St. Patrick's Church, Patrington
  dir=$RIR_home_abs/open_air/stpatricks_church/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington/stereo/ortf_s1r1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington/stereo/ortf_s2r2.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington/stereo/ortf_s3r3.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington/b-format/soundfield_s1r1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington/b-format/soundfield_s2r2.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington/b-format/soundfield_s3r3.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  #St. Patrick's Church, Patrington - Model
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington-model/b-format/stpatricks_s1r1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington-model/b-format/stpatricks_s2r2.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-patricks-church-patrington-model/b-format/stpatricks_s3r3.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #St. Mary's Abbey Reconstruction
  dir=$RIR_home_abs/open_air/stmarys_abbey_reconstruction/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/stereo/phase1_stereo.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/stereo/phase2_stereo.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/stereo/phase3_stereo.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/b-format/phase1_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/b-format/phase2_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/b-format/phase3_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/st-marys-abbey-reconstruction/b-format/phase1_bformat_catt.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Cafeteria Universidad San Buenaventura Bogotá 
  dir=$RIR_home_abs/open_air/cafeteria_universidad_san_buenaventura_bogota/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/leorodino/cafeteria-universidad-san-buenaventura-bogota/mono/m0003_s01_r01_1.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #York Minster
  dir=$RIR_home_abs/open_air/york_minster/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/york-minster/stereo/minster1_000_ortf_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/york-minster/b-format/minster1_bformat_48k.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #WNIU Studio Untreated
  dir=$RIR_home_abs/open_air/wniu_studio_untreated/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/jaddie/wniu-studio-untreated/mono/3.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  #Gill heads mine
  dir=$RIR_home_abs/open_air/gill_heads_mine/
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/mono/mine_site1_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/mono/mine_site2_1way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/mono/mine_site1_2way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/mono/mine_site2_2way_mono.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/b-format/mine_site1_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/b-format/mine_site2_1way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/b-format/mine_site1_2way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh
  echo "wget http://www.openairlib.net/sites/default/files/auralization/data/audiolab/gill-heads-mine/b-format/mine_site2_2way_bformat.wav --directory-prefix=$dir " >> $log_dir/${DBname}_download_commands.sh

  rm -rf $RIR_home/open_air
  download_file=$log_dir/${DBname}_download_commands.sh
  if [ ! -z "$file_splitter" ]; then
    num_jobs=$($file_splitter $download_file || exit 1)
    job_file=${download_file%.sh}.JOB.sh
    job_log=${download_file%.sh}.JOB.log
  else
    num_jobs=1
    job_file=$download_file
    job_log=${download_file%.sh}.log
  fi
  # execute the commands using the above created array jobs
  time $decode_cmd --max-jobs-run 40 JOB=1:$num_jobs $job_log \
    sh $job_file || exit 1;

fi

# reformat the downloaded RIRs
command_file=$log_dir/${DBname}_read_rir_noise.sh
echo "">$command_file

type_num=1
data_files=( $(find $RIR_home/open_air/ -name '*.wav' -type f -print || exit -1) )
total_files=$(echo ${data_files[@]}|wc -w)
echo "" > $log_dir/${DBname}_type${type_num}.rir.list
echo "Found $total_files impulse responses in ${RIR_home}/open_air/"
file_count=1 # affix to ensure that files with same name are not overwritten
for data_file in ${data_files[@]}; do
  # open-air has multiple formats of wav audio, some of which are not compatible with python's wav.read() function
#  output_file_name=${DBname}_type${type_num}_${file_count}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  output_file_name=${DBname}_type${type_num}_`basename $data_file| tr '[:upper:]' '[:lower:]'`
  echo "sox -t wav $data_file -t wav -r $sampling_rate -e signed-integer -b $output_bit ${output_dir}/${output_file_name}" >> $command_file
  echo ${output_dir}/${output_file_name} >>  $log_dir/${DBname}_type${type_num}.rir.list
  file_count=$((file_count + 1))
done


if [ ! -z "$file_splitter" ]; then
  num_jobs=$($file_splitter $command_file || exit 1)
  job_file=${command_file%.sh}.JOB.sh
  job_log=${command_file%.sh}.JOB.log
else
  num_jobs=1
  job_file=$command_file
  job_log=${command_file%.sh}.log
fi
# execute the commands using the above created array jobs
time $decode_cmd --max-jobs-run 40 JOB=1:$num_jobs $job_log \
  sh $job_file || exit 1;

