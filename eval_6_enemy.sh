db_path='/workspace/eval_6_enemy.db'
for i in 1 2 3
do
    echo -----------------------------------------------------------------------
    python eval_6_enemy.py --exper-name switch-$i --model-path ckpt/rl-$i.pt --coach-name rule-based --adv-coach --save-to-db $db_path
    echo -----------------------------------------------------------------------
    for name in red rl joint airl
    do
        echo -----------------------------------------------------------------------
        python eval_6_enemy.py --exper-name $name-$i --model-path ckpt/$name-$i.pt --coach-name rule-based --adv-coach --save-to-db $db_path
        echo -----------------------------------------------------------------------
    done
done
for dropout in 0 1
do
    for i in 1 2 3
    do
        echo -----------------------------------------------------------------------
        python eval_6_enemy.py --exper-name switch-$i --model-path ckpt/rl-$i.pt --coach-name rule-based --dropout $dropout --save-to-db $db_path
        echo -----------------------------------------------------------------------
        for name in red rl joint airl
        do
            echo -----------------------------------------------------------------------
            python eval_6_enemy.py --exper-name $name-$i --model-path ckpt/$name-$i.pt --coach-name rule-based --dropout $dropout --save-to-db $db_path
            echo -----------------------------------------------------------------------
        done
    done
done