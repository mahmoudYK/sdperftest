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

## Examples:
Generate and enable 1000 services of type DAG with edge probabilty 0.01:
```sh
$ sudo python3 generate_services.py  -t dag -p 0.01 -n 1000 -g
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

run load test based on a test services generator, the results are printed to a json file 
and a plot is generated for the data collected from 2 different systemd binaries. 

## Packages need to be installed:
matplotlib: https://matplotlib.org/stable/users/installing/index.html

## Options:
-  -h, --help  
                        show this help message and exit
-  -d SYSTEMD_BIN_PATH, --systemd_bin_path SYSTEMD_BIN_PATH  
                        list of systemd bin directories
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

## Examples:
Generate this sequence of 'dag' services numbers with edge probability=0.2 [500 1000 1500 2000 2500].
Use EUID and EGID of 1000 to run 'systemd --test --system --no-pager' command for each number of 
generated services. The output files should be named results.json and results.jpg.
```sh
$ sudo python3 run_tests.py -t dag -p .2 -s 500 -j 500 -n 5 -d $SD_BIN_PATH1/systemd -d $SD_BIN_PATH2/systemd -o results -u 1000 -g 1000
```

## License:
LGPL-2.1-or-later
