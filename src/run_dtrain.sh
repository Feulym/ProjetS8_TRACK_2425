#!/bin/bash
# Liste des machines (assurez-vous que votre clé SSH est configurée pour chaque machine)
MACHINES=("machine1" "machine2" "machine3")

# Dossier du projet sur la machine distante
REMOTE_DIR="/chemin/vers/votre/projet"
EXEC="nom_executable.ext"

# Hyperparamètres fixes
EPOCHS=50
BATCH_SIZE=32

# Hyperparamètres à varier
LEARNING_RATES=(1e-3 1e-4 1e-5)
HIDDEN_DIMS=(32 64 128 256)
NUM_LAYERS=(1 2 3)

# Nombre total de machines
TOTAL_MACHINES=${#MACHINES[@]}
i=0

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
        for NUM_LAYER in "${NUM_LAYERS[@]}"; do
            MACHINE=${MACHINES[$(( i % TOTAL_MACHINES ))]}
            
            echo "Lancement sur $MACHINE avec learning_rate=$LEARNING_RATE, hidden_dim=$HIDDEN_DIM, num_layers=$NUM_LAYER"
            
            ssh "$MACHINE" "cd $REMOTE_DIR && source .env/bin/activate && python $EXEC \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --hidden_dim $HIDDEN_DIM \
                --num_layers $NUM_LAYER \
                --output_dir results" &
            
            i=$(( i + 1 ))
        done
    done
done

wait
echo "Tous les entraînements ont été lancés."