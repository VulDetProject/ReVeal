# Static Code Slicer for C Code #

Intra-function slicer for C code. This repository a lightweight implementation of extracting forward and backward slices
from a given source code. 

This slicer is based on [Joern Fuzzy Code parser](https://github.com/octopus-platform/joern/). 

## Depndencies

1. Gradle 2.0, which further requires Java 8 (`sudo apt-get install openjdk-8-jdk`)
```
wget https://services.gradle.org/distributions/gradle-2.0-bin.zip
sudo mkdir /opt/gradle
sudo unzip gradle-2.0-bin.zip -d /opt/gradle
echo 'export PATH="$PATH:/opt/gradle/gradle-2.0/bin"' > ~/.bashrc
source ~/.bashrc
```
3. Graphviz (`sudo apt install graphviz-dev`)
4. Python >= 3.5 with graphviz library (`pip install graphviz`)

## Installation
1. Enter into the source code
    ```
    cd code-slicer
   ```
2. Build **Joern**
    ```
   cd joern
   ./build.sh
   cd ..
   ```

## Usage
To extract slice use
 ```
    ./slicer.sh <Dir> <FileName> <LineNo> <OutDir> <DataFlowOnly(Optional)>
 ```

For Example
 ```
    ./slicer.sh test test1.c 7 test-output
 ```

##### Arguments
1. Dir: The directory path of the .C file.
2. FileName: Name of the C file (*.c format)
3. LineNo: Line Number where the slice will begin.
4. OutDir: The Directory path for the output. 
5. DataFlowOnly: This is optional, if this is mentioned, slicer will produce slices only on dataflow graph.

## Output
This slicer will produce 4 outputs in the **OutDir**. 

- __\<OutDir>/\<FileName\>.forward__ will contain the lines in the forward slice.
- __\<OutDir>/\<FileName\>.backward__ will contain the lines in the backward slice.
- __\<OutDir>/\<FileName\>pdf__ will contain the visual representation of the graph.
- __\<OutDir>/\<FileName\>__ will contain the dot representation of the graph. 

### Limitations
- As of now, we assume only one function per c file. 

 
