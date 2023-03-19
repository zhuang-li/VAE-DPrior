for shuffle in {0..4}
do
  for shot in 0 1 5
  do
    if [ ${shot} != 0 ]; then
      python style_transfer.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/empathetic/optimus/finetune.json
      python generate_optimus_examples.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/empathetic/optimus/test.json
    else
      python generate_optimus_examples.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/empathetic/optimus/test.json
    fi
  done
done