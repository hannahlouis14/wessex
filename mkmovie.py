import os
os.system("ffmpeg -f image2 -pattern_type glob -r 3 -i '/mnt/storage6/hlouis/plots/figures/sea_ice/thickness/*EPM152*.png' -vcodec mpeg4 -y /mnt/storage6/hlouis/plots/figures/sea_ice/Sea_Ice_thickness_ANHA4_EPM152_speed3.mp4");
