for k in 1 2 3
do
    python3 main0.py --execute 'F-MCdrop-entropy' --gpu 0 --K $k
done