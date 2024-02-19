echo "##############################################################################"
echo "[Main] Start NeurIPS 2024 - experiment"
echo
start_time=`date +%s.%N`

echo "##############################################################################"
echo "[Main] Sandbox"
python3 0_sandbox.py
echo

echo "##############################################################################"
echo "[Main] Cropping figures"
for FILE in figures/*.pdf; do
  pdfcrop --noverbose "${FILE}" "${FILE}"
done
echo

storage_dir='/mnt/d/'
if [ -d $storage_dir ]; then
  echo "##############################################################################"
  printf "[Main] Sending figures to %s \n" $storage_dirx
  cp -vr figures/ $storage_dir
  echo
fi

echo "##############################################################################"
end_time=`date +%s.%N`
runtime=$(echo "$end_time - $start_time" | bc)
printf "[Main] NeurIPS 2024 - all experiment done in %.1f seconds\n" $runtime
echo
