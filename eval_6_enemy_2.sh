db_path='/workspace/eval_6_enemy.db'
for i in 1 2 3
do
    echo -----------------------------------------------------------------------
    python eval_6_enemy.py --exper-name switch-$i --model-path ckpt/rl-$i.pt --coach-name random --p 1 --save-to-db $db_path
    echo -----------------------------------------------------------------------
    for name in red rl joint airl
    do
        echo -----------------------------------------------------------------------
        python eval_6_enemy.py --exper-name $name-$i --model-path ckpt/$name-$i.pt --coach-name random --p 1 --save-to-db $db_path
        echo -----------------------------------------------------------------------
    done
done
for i in 1 2 3
do
    echo -----------------------------------------------------------------------
    python eval_6_enemy.py --exper-name switch-$i --model-path ckpt/rl-$i.pt --save-to-db $db_path
    echo -----------------------------------------------------------------------
    for name in red rl joint airl
    do
        echo -----------------------------------------------------------------------
        python eval_6_enemy.py --exper-name $name-$i --model-path ckpt/$name-$i.pt --save-to-db $db_path
        echo -----------------------------------------------------------------------
    done
done