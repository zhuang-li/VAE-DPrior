python style_transfer.py --config style_transfer/config/persona/optimus/pre_train.json
for shuffle in {0..4}
do
  for shot in 0 1 5
  do
    if [ ${shot} != 0 ]; then
      python style_transfer.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/persona/optimus/finetune.json
      python generate_optimus_examples.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/persona/optimus/test.json
    else
      python generate_optimus_examples.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/persona/optimus/test.json
    fi
  done
done