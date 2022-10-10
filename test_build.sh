for cmd in 4 0 1 2 3 5
do
    for i in 1 2 3
    do
        echo ------------------------------------------------------------------------------------
        python test_build.py --exper-name switch-$i --model-path ckpt/rl-$i.pt --cmd-idx $cmd --save-to-db /workspace/test_build.db
        echo ------------------------------------------------------------------------------------
    done
    for model in red rl joint airl
    do
        for i in 1 2 3
        do
            echo ------------------------------------------------------------------------------------
            python test_build.py --exper-name $model-$i --model-path ckpt/$model-$i.pt --cmd-idx $cmd --save-to-db /workspace/test_build.db
            echo ------------------------------------------------------------------------------------
        done
    done
done