# Collection of scripts for systemd load test and performance analysis.

## generate_services.py:

Generator of systemd test services with different kinds of dependencies, currently the supported dependecies are:
- [parallel]: basic services without any dependencies among each other.
- [single_path]: each service depends only on the previous service, 
except test_single_path0.service which is considered the root of the single path graph.
so, it doesn't depend on any services 
- [dag]: directed acyclic graph of random service dependeencies with edge probability option to control the
amount of possible edges (dependencies) between services.

## Packages needs to be installed:
pystemd: install instructions can be found at https://github.com/facebookincubator/pystemd 

## Options:
-  -h, --help  
                        show this help message and exit.
-  -n NUM_OF_SERVICES, --num_of_services NUM_OF_SERVICES  
                        number of services to generate, if not used, 500 is default.
-  -t TYPE, --type TYPE  
                        type of generated test services as comma separated list [parallel,single_path,dag].
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

## License:
LGPL-2.1-or-later
