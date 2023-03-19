for shuffle in {0..4}
do
  for shot in 0 1 5
  do
    python generate_casual_examples.py --shuffle ${shuffle} --shot ${shot} --config style_transfer/config/persona/casual/test.json
  done
done