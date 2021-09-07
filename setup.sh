#!/bin/bash
# @author: Victor Faraggi
# This script provides an automatic way of downloading and uploading protein stability stored on Dropbox
# It seeks to allow for a fast and up-to-date setup on different machines.
#
# Arguments :
#	-u / --upload   : Uploads the data folders in parallel-synthesis/ and mutagens/ to Dropbox.
#       -d / --download : Downloads the data from a hard-coded Dropbox url. It also uncompresses, moves and cleans the files.
# 	-h / --help     : Displays this information.
#
# Examples :
#	./setup.sh              ---> It will download data and ask you to setup a Dropbox uploader conf.
#       ./setup.sh --upload     ---> It will only upload the data folders.
#       ./setup.sh --download   ---> It will only download the data.


############# DROPBOX SETUP #############
DROPBOX_CONF=$HOME/.dropbox_uploader

function setup_dropbox {
  if test -f "$DROPBOX_CONF"; then
    echo "Dropbox parece estar configurado. Chequea $DROPBOX_CONF";
    return 0;
  else
    wget https://raw.githubusercontent.com/andreafabrizi/Dropbox-Uploader/master/dropbox_uploader.sh;
    chmod +x dropbox_uploader.sh;
    bash dropbox_uploader.sh;
  fi
  return 1;
}

############# DOWNLOAD DATA #############
function download_data {
  wget https://www.dropbox.com/sh/ifindwj2hlmyo6f/AADFUJFR_9OWMnctXckQvmlxa\?dl\=0 -O "data-stability.zip";
}

function setup_data {
  # download_data;
  wget https://www.dropbox.com/sh/ifindwj2hlmyo6f/AADFUJFR_9OWMnctXckQvmlxa\?dl\=0 -O "data-stability.zip";
  unzip data-stability.zip;
  tar -xvf data-parallel.tar.gz;
  tar -xvf data-mutagenesis.tar.gz;
  tar -xvf data-fireprot.tar.gz;
  rm data-stability.zip data-mutagenesis.tar.gz data-parallel.tar.gz data-fireprot.tar.gz;
  return  1;
}

############# UPLOAD DATA #############
function upload_data {
  while true; do
    if test -f "$DROPBOX_CONF"; then
      tar --exclude "project/parallel_synthesis/data/raw/*" \
          --exclude "project/parallel_synthesis/data/mmseq_*" \
          --exclude "project/parallel_synthesis/data/tmp" \
          -czvf data-parallel.tar.gz project/parallel_synthesis/data/;
      tar --exclude "project/mutagenesis/data/Protera/mmseq_*" \
          --exclude "project/mutagenesis/data/Protera/raw" \
          --exclude "project/mutagenesis/data/Protera/tmp" \
          --exclude "project/mutagenesis/data/Protera/prism" \
          -czvf data-mutagenesis.tar.gz project/mutagenesis/data/Protera;
      tar -czvf data-fireprot.tar.gz project/fireprot/data/;
      bash dropbox_uploader.sh upload data-mutagenesis.tar.gz protera-data/;
      bash dropbox_uploader.sh upload data-parallel.tar.gz protera-data/;
      bash dropbox_uploader.sh upload data-fireprot.tar.gz protera-data/;
      rm data-mutagenesis.tar.gz data-parallel.tar.gz data-fireprot.tar.gz;
      break;
    else
      setup_dropbox;
    fi
  done
  return 1;
}

############# HELP #############
function display_help {
	echo "setup.sh"
	echo "@author: Victor Faraggi"
	echo "This script provides an automatic way of downloading and uploading protein stability stored on Dropbox"
	echo "It seeks to allow for a fast up-to-date setup on different machines."
	echo
	echo "Arguments :"
        echo "  -u / --upload   : Uploads the data folders in parallel-synthesis/ and mutagens/ to Dropbox."
        echo "  -d / --download : Downloads the data from a hard-coded Dropbox url. It also uncompresses, moves and cleans the files."
	echo "	-h / --help     : Displays this information."
	echo
	echo "Examples :"
        echo "	./setup.sh              ---> It will download data, set it up, and ask you to setup a Dropbox uploader conf."
	echo "	./setup.sh --upload     ---> It will only upload the data folders."
        echo "	./setup.sh --download   ---> It will only download the data."
	echo
}



############# MAIN PROGRAM  #############
function main {
  while (( "$#" )); do
    echo "$1";
    case "$1" in
      -u|--upload)
        echo "Performing upload..."; sleep 1;
        upload_data;
        exit 0
        ;;
      -d|--download)
        echo "Performing download only..."; sleep 1;
        download_data;
        exit 0
        ;;
    esac
  done

  # TODO: integrate this in the previous while/case 
  echo "Performing setup..."; sleep 1;
        setup_data;
        setup_dropbox;
        exit 0;
  return 1;
}

main