echo "##############################################################################"
echo "[Main] Start NeurIPS 2024 - experiment"
echo
start_time=`date +%s.%N`

echo "##############################################################################"
echo "[Main] Experiment: 'Sandbox'"
python3 -W ignore 0_sandbox.py
echo

echo "##############################################################################"
echo "[Main] Cropping figures"
for FILE in _figures_async_players/*.pdf; do
  pdfcrop --noverbose "${FILE}" "${FILE}"
done
echo

storage_dir='/mnt/d'
if [ -d $storage_dir ]; then
  echo "##############################################################################"
  printf "[Main] Sending figures to %s \n" $storage_dir
  cp -vr _figures_async_players/ $storage_dir
  echo
fi

echo "##############################################################################"
end_time=`date +%s.%N`
runtime=$(echo "$end_time - $start_time" | bc)
printf "[Main] NeurIPS 2024 - all experiment done in %.1f seconds\n" $runtime
echo
