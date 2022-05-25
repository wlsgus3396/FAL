for k in 1 2 3
do
    python3 main.py --execute 'F-MCdrop-entropy' --gpu 1 --K $k
done