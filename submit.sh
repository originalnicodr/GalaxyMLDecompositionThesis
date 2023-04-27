#!/bin/bash

### Las líneas #SBATCH configuran los recursos de la tarea
### (aunque parezcan estar comentadas)

### Nombre de la tarea
#SBATCH --job-name=linkage-analysis

#SBATCH   --mail-type=ALL
#SBATCH   --mail-user=niconavall@gmail.com

### Tiempo de ejecucion. Formato dias-horas:minutos.
#SBATCH --time 3-0:00

### Numero de procesos a ser lanzados.
#SBATCH --ntasks=112
#SBATCH --nodes=2

### Nombre de partcion
#SBATCH --partition=batch

### Script que se ejecuta al arrancar el trabajo

### Cargar el entorno del usuario incluyendo la funcionalidad de modules
### No tocar
. /etc/profile

### Cargar los módulos para la tarea
# FALTA: Agregar los módulos necesarios
module load gcc
module load openmpi
### Largar el programa con los cores detectados
### La cantidad de nodos y cores se detecta automáticamente
srun source activate gchop-env && python src/galaxychop/assert-ward-linkage.py > slurm_output.txt