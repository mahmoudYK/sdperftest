#!/bin/bash
# License: LGPL-2.1-or-later

set -e

EXIT_SUCCESS=0
EXIT_ERROR=1
INITIAL_WD=$(pwd)

usage() {
	echo "**************************************************"
	echo "Clone & build systemd using a specific"
    echo "commit hash: https://github.com/systemd/systemd"
	echo "Options:"
	echo "\$1  systemd build dir"
	echo "\$2  systemd commit hash"
	echo "**************************************************"
}


# print failure msg and exit with $EXIT_ERROR.
# args:
# $1: failure msg 
fail() {
    echo "error: $1"
    cd $INITIAL_WD
    exit $EXIT_ERROR
}


if [ ! $# -eq 2 ]; then
	if [ $1 == "-h" ]; then
		usage
		exit $EXIT_SUCCESS
	else
		usage
		fail "wrong number of arguments!"
	fi
fi


BUILD_DIR=$1
COMMIT_HASH=$2
SYSTEMD_SOURCE_PATH=${BUILD_DIR}/${COMMIT_HASH}/systemd
SYSTEMD_GIT_REPO="https://github.com/systemd/systemd"
MESON_BULD_DIR=build

# clone git repo to a certain directory.
# args:
# $1: git repo to clone
# $2: directory 
# return:
# exit value of the git clone command  
clone_git_repo() {
    echo "clone : $1"
    git clone $1 $2
    return $?
}


# checkout a certain commit hash.
# (NOTE: needs to handle "You are in 'detached HEAD' state")
# args:
# $1: commit hash
# return:
# exit value of the git checkout command  
checkout_commit_hash() {
    echo "checkout commit hash: $1"
    git checkout $1
    return $?
}


# build systemd from source code using meson build system and ninja as a backend.
# args:
# $1: meson build directory
# return:
# exit value of the ninja -C command
meson_build_sd() {
    if [ ! -f ${1}/build.ninja ] ; then
         meson $1 \
                -D man=false \
                -D translations=false \
                -D mode=developer
    fi
    echo $(pwd)
    ninja -C $1
    return $? 
}

   
if [ -x  ${SYSTEMD_SOURCE_PATH}/build/systemd ]; then
    echo "systemd executbale is already existing, no need to build"
    exit $EXIT_SUCCESS

elif [ ! -d  ${SYSTEMD_SOURCE_PATH} ]; then
    mkdir -p  ${SYSTEMD_SOURCE_PATH}

else
    echo "${SYSTEMD_SOURCE_PATH} is already existing!"     
fi

cd  ${SYSTEMD_SOURCE_PATH}

# check if $SYSTEMD_SOURCE_PATH is empty,
# if it is empty, then clone from $SYSTEMD_GIT_REPO,
# checkout $COMMIT_HASH, and start meson build. 
# if it is not empty then assume that systemd repo is ok!
# and start the meson build process. 
if [ -z "$(ls -A)" ]; then   
    ! clone_git_repo ${SYSTEMD_GIT_REPO} . && fail "clonning ${SYSTEMD_GIT_REPO} failed!"   
    ! checkout_commit_hash ${COMMIT_HASH} && fail "checking out ${COMMIT_HASH} failed!"       
fi

meson_build_sd ${MESON_BULD_DIR}
ret=$?

cd $INITIAL_WD

exit $ret