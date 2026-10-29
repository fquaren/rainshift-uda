OUTPUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/results_rainshift_uda"

cd $OUTPUT_DIR/unet
for src in europe_west horn-of-africa melanesia; do
  for tgt in europe_west horn-of-africa melanesia; do
    [ "$src" = "$tgt" ] && continue
    cp -r ${src}__to__${src}__none ${src}__to__${tgt}__none
  done
done
