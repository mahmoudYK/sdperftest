# Collection of scripts for systemd load test and performance analysis.

## generate_services.py:

Generator of systemd test services with different kinds of dependencies, currently the supported dependecies are:
- [parallel]: basic services without any dependencies among each other.
- [single_path]: each service depends only on the previous service, 
except test_single_path0.service which is considered the root of the single path graph.
so, it doesn't depend on any services. 
- [dag]: directed acyclic graph of random service dependeencies with edge probability option to control the
amount of possible edges (dependencies) between services.

## Packages need to be installed:
pystemd: https://github.com/facebookincubator/pystemd  
graphviz: https://pypi.org/project/graphviz/ 

## Options:
-  -h, --help  
                        show this help message and exit.
-  -n NUM_OF_SERVICES, --num_of_services NUM_OF_SERVICES  
                        number of services to generate, if not used, 500 is default.
-  -t TYPE, --type TYPE  
                        type of generated test services [parallel,single_path,dag].
-  -g, --generate  
                        generate and enable all the test services (default behaviour if -r is not used).
-  -r, --remove  
                        disbale and remove all the test services.
-  -p EDGE_PROBABILITY, --edge_probability EDGE_PROBABILITY  
                        edge probability for DAG services generator (default: 0.1).
-  -a, --analyze  
                        analyze systemd boot time.
-  -z, --gen_graphviz_dot  
                        generate graphviz dot file
-  -d DOT_DIR, --dot_dir DOT_DIR  
                        graphviz dot file output directory


## Examples:
Generate and enable 1000 services of type DAG with edge probabilty 0.01,
also generate graphviz dot source file and save it to dot_dir directory:
```sh
$ sudo python3 generate_services.py  -t dag -p 0.01 -n 1000 -g -z -d dot_dir
```
(-g an be omitted)

Disable and remove all services of type DAG:
```sh
$ sudo python3 generate_services.py  -t dag -r
```
Generate and enable 600 services of type parallel and 1000 services of type single_path:
```sh
$ sudo python3 generate_services.py  -t parallel -n 600 -t single_path -n 1000 
```
Disable and remove all services of type parallel and single_path:
```sh
$ sudo python3 generate_services.py  -t parallel -t single_path -r
```
Generate 1000 services of type DAG, 1000 of type parallel and 1000 of type single_path:
```sh
$ sudo python3 generate_services.py  -t dag -t parallel -t single_path -n 1000
```
Analyze boot time of systemd:
```sh
$ sudo python3 generate_services.py -a
```

## run_sd_load_tests.py

Run load test based on a test services generator, the results are printed to a json file 
and a plot is generated for the data collected from 2 different systemd binaries. 

## Packages need to be installed:
matplotlib: https://matplotlib.org/stable/users/installing/index.html

## Options:
-h, --help  
                        show this help message and exit
-  -m SD_PATH_MODE, --sd_path_mode SD_PATH_MODE  
                        use 2 systemd executable paths or 2 systemd upstream commit hashes [exe,commit]
-  -e SD_REF_EXE_PATH, --sd_ref_exe_path SD_REF_EXE_PATH  
                        reference systemd executable path
-  -f SD_COMP_EXE_PATH, --sd_comp_exe_path SD_COMP_EXE_PATH  
                        compared systemd executable path
-  -c SD_COMMIT_REF, --sd_commit_ref SD_COMMIT_REF  
                        commit hash of the reference systemd repo
-  -d SD_COMMIT_COMP, --sd_commit_comp SD_COMMIT_COMP  
                        commit hash of the compared systemd repo
-  -t GEN_TYPES, --gen_types GEN_TYPES  
                        type of generated test services [parallel,single_path,dag]
-  -p DAG_EDGE_PROBABILITY, --dag_edge_probability DAG_EDGE_PROBABILITY  
                        edge probability for DAG services generator
-  -s START_SERVICES_NUM, --start_services_num START_SERVICES_NUM  
                        number of generated services to start the first test
-  -j STEP_SERVICES_NUM, --step_services_num STEP_SERVICES_NUM  
                        number of generated services added to the previouse test to jump to the next test
-  -n TESTS_NUM, --tests_num TESTS_NUM  
                        number of tests to run
-  -u USER_UID, --user_uid USER_UID  
                        user UID to run systemd in test mode
-  -g USER_GID, --user_gid USER_GID  
                        user GID to run systemd in test mode
-  -o OUTPUT_FILES_NAME, --output_files_name OUTPUT_FILES_NAME  
                        name of output json data and jpeg plot files
-  -r OUTPUT_FILES_DIR, --output_files_dir OUTPUT_FILES_DIR  
                        output artifacts dir
-  -z, --gen_graphviz_dot  
                        generate graphviz dot file
-  -l, --gen_flamegraph  
                        generate per test flamegraph
-  -F PERF_FREQUENCY, --perf_frequency PERF_FREQUENCY  
                        perf profiling frequency
-  -S PERF_SLEEP_PERIOD, --perf_sleep_period PERF_SLEEP_PERIOD  
                        perf command sleep time before stop recording


## Examples:
Generate this sequence of 'dag' services numbers (with edge probability=0.2): [500 1000 1500 2000 2500].
Use EUID and EGID of 1000 to run 'systemd --test --system --no-pager' command for each number of 
generated services. The output files should be named results.json and results.jpg.
```sh
$ sudo python3 run_tests.py -t dag -p .2 -s 500 -j 500 -n 5 -m exe -e $SD_BIN_PATH1/systemd -f $SD_BIN_PATH2/systemd -o results -r $OUTPUT_DIR -u 1000 -g 1000
```
Generate the same sequence of services as in the previous example, however use systemd git commit hashes instead of executable paths, 
and generate graphviz dot files, save the dot files at $OUTPUT_DIR 
```sh
$ sudo python3 run_tests.py -t dag -p .2 -s 500 -j 500 -n 5 -m commit -c $SD_COMMIT_HASH1 -d $SD_COMMIT_HASH2 -o results -r $OUTPUT_DIR -u 1000 -g 1000 -z
```

## build_sd.sh

Clone & build systemd using a specific git commit hash from systemd github repo: https://github.com/systemd/systemd

## Options:
-  -h  
                    print usage and exit
-  -d  
                    systemd build dir
-  -c  
                    systemd git commit hash
## Examples:
```sh
$ ./build_sd.sh -d $SD_BUILD_DIR -c $SD_COMMIT_HASH
```

## License:
LGPL-2.1-or-later
