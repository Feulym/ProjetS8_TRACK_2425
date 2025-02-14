#!/bin/bash
# Liste des machines (assurez-vous que votre clé SSH est configurée pour chaque machine)
MACHINES=("machine1" "machine2" "machine3")

# Dossier du projet sur la machine distante
REMOTE_DIR="/chemin/vers/votre/projet"

# Hyperparamètres
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=0.001

HIDDEN_DIMS=(32 64 128 256)
NUM_LAYERS=(1 2 3)

for MACHINE in "${MACHINES[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
        for NUM_LAYER in "${NUM_LAYERS[@]}"; do
            echo "Lancement sur $MACHINE avec hidden_dim=$HIDDEN_DIM, num_layers=$NUM_LAYER"
            ssh "$MACHINE" "cd $REMOTE_DIR && source .env/bin/activate && python lstm_train.py \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --hidden_dim $HIDDEN_DIM \
                --num_layers $NUM_LAYER \
                --output_dir results" &
        done
    done
done

wait
echo "Tous les entraînements ont été lancés."
