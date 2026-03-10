#!/usr/bin/env bash
BASE_URL="https://dcapswoz.ict.usc.edu/wwwedaic/data"
DEST="$(cd "$(dirname "$0")" && pwd)"
LOG="$DEST/download.log"
PARALLEL=4

FILES=(
300_P.tar.gz 301_P.tar.gz 302_P.tar.gz 303_P.tar.gz 304_P.tar.gz
305_P.tar.gz 306_P.tar.gz 307_P.tar.gz 308_P.tar.gz 309_P.tar.gz
310_P.tar.gz 311_P.tar.gz 312_P.tar.gz 313_P.tar.gz 314_P.tar.gz
315_P.tar.gz 316_P.tar.gz 317_P.tar.gz 318_P.tar.gz 319_P.tar.gz
320_P.tar.gz 321_P.tar.gz 322_P.tar.gz 323_P.tar.gz 324_P.tar.gz
325_P.tar.gz 326_P.tar.gz 327_P.tar.gz 328_P.tar.gz 329_P.tar.gz
330_P.tar.gz 331_P.tar.gz 332_P.tar.gz 333_P.tar.gz 334_P.tar.gz
335_P.tar.gz 336_P.tar.gz 337_P.tar.gz 338_P.tar.gz 339_P.tar.gz
340_P.tar.gz 341_P.tar.gz 343_P.tar.gz 344_P.tar.gz 345_P.tar.gz
346_P.tar.gz 347_P.tar.gz 348_P.tar.gz 349_P.tar.gz 350_P.tar.gz
351_P.tar.gz 352_P.tar.gz 353_P.tar.gz 354_P.tar.gz 355_P.tar.gz
356_P.tar.gz 357_P.tar.gz 358_P.tar.gz 359_P.tar.gz 360_P.tar.gz
361_P.tar.gz 362_P.tar.gz 363_P.tar.gz 364_P.tar.gz 365_P.tar.gz
366_P.tar.gz 367_P.tar.gz 368_P.tar.gz 369_P.tar.gz 370_P.tar.gz
371_P.tar.gz 372_P.tar.gz 373_P.tar.gz 374_P.tar.gz 375_P.tar.gz
376_P.tar.gz 377_P.tar.gz 378_P.tar.gz 379_P.tar.gz 380_P.tar.gz
381_P.tar.gz 382_P.tar.gz 383_P.tar.gz 384_P.tar.gz 385_P.tar.gz
386_P.tar.gz 387_P.tar.gz 388_P.tar.gz 389_P.tar.gz 390_P.tar.gz
391_P.tar.gz 392_P.tar.gz 393_P.tar.gz 395_P.tar.gz 396_P.tar.gz
397_P.tar.gz 399_P.tar.gz 400_P.tar.gz 401_P.tar.gz 402_P.tar.gz
403_P.tar.gz 404_P.tar.gz 405_P.tar.gz 406_P.tar.gz 407_P.tar.gz
408_P.tar.gz 409_P.tar.gz 410_P.tar.gz 411_P.tar.gz 412_P.tar.gz
413_P.tar.gz 414_P.tar.gz 415_P.tar.gz 416_P.tar.gz 417_P.tar.gz
418_P.tar.gz 419_P.tar.gz 420_P.tar.gz 421_P.tar.gz 422_P.tar.gz
423_P.tar.gz 424_P.tar.gz 425_P.tar.gz 426_P.tar.gz 427_P.tar.gz
428_P.tar.gz 429_P.tar.gz 430_P.tar.gz 431_P.tar.gz 432_P.tar.gz
433_P.tar.gz 434_P.tar.gz 435_P.tar.gz 436_P.tar.gz 437_P.tar.gz
438_P.tar.gz 439_P.tar.gz 440_P.tar.gz 441_P.tar.gz 442_P.tar.gz
443_P.tar.gz 444_P.tar.gz 445_P.tar.gz 446_P.tar.gz 447_P.tar.gz
448_P.tar.gz 449_P.tar.gz 450_P.tar.gz 451_P.tar.gz 452_P.tar.gz
453_P.tar.gz 454_P.tar.gz 455_P.tar.gz 456_P.tar.gz 457_P.tar.gz
458_P.tar.gz 459_P.tar.gz 461_P.tar.gz 462_P.tar.gz 463_P.tar.gz
464_P.tar.gz 465_P.tar.gz 466_P.tar.gz 467_P.tar.gz 468_P.tar.gz
469_P.tar.gz 470_P.tar.gz 471_P.tar.gz 472_P.tar.gz 473_P.tar.gz
474_P.tar.gz 475_P.tar.gz 476_P.tar.gz 477_P.tar.gz 478_P.tar.gz
479_P.tar.gz 480_P.tar.gz 481_P.tar.gz 482_P.tar.gz 483_P.tar.gz
484_P.tar.gz 485_P.tar.gz 486_P.tar.gz 487_P.tar.gz 488_P.tar.gz
489_P.tar.gz 490_P.tar.gz 491_P.tar.gz 492_P.tar.gz
600_P.tar.gz 601_P.tar.gz 602_P.tar.gz 603_P.tar.gz 604_P.tar.gz
605_P.tar.gz 606_P.tar.gz 607_P.tar.gz 608_P.tar.gz 609_P.tar.gz
612_P.tar.gz 615_P.tar.gz 617_P.tar.gz 618_P.tar.gz 619_P.tar.gz
620_P.tar.gz 622_P.tar.gz 623_P.tar.gz 624_P.tar.gz 625_P.tar.gz
626_P.tar.gz 627_P.tar.gz 628_P.tar.gz 629_P.tar.gz 631_P.tar.gz
632_P.tar.gz 633_P.tar.gz 634_P.tar.gz 635_P.tar.gz 636_P.tar.gz
637_P.tar.gz 638_P.tar.gz 640_P.tar.gz 641_P.tar.gz 649_P.tar.gz
650_P.tar.gz 651_P.tar.gz 652_P.tar.gz 653_P.tar.gz 654_P.tar.gz
655_P.tar.gz 656_P.tar.gz 657_P.tar.gz 658_P.tar.gz 659_P.tar.gz
660_P.tar.gz 661_P.tar.gz 662_P.tar.gz 663_P.tar.gz 664_P.tar.gz
666_P.tar.gz 667_P.tar.gz 669_P.tar.gz 670_P.tar.gz 673_P.tar.gz
676_P.tar.gz 677_P.tar.gz 679_P.tar.gz 680_P.tar.gz 682_P.tar.gz
683_P.tar.gz 684_P.tar.gz 687_P.tar.gz 688_P.tar.gz 689_P.tar.gz
691_P.tar.gz 692_P.tar.gz 693_P.tar.gz 695_P.tar.gz 696_P.tar.gz
697_P.tar.gz 698_P.tar.gz 699_P.tar.gz 702_P.tar.gz 703_P.tar.gz
705_P.tar.gz 707_P.tar.gz 708_P.tar.gz 709_P.tar.gz 710_P.tar.gz
712_P.tar.gz 713_P.tar.gz 715_P.tar.gz 716_P.tar.gz 717_P.tar.gz
718_P.tar.gz
)

TOTAL=${#FILES[@]}
echo "[$(date)] Starting download of $TOTAL files to $DEST" | tee "$LOG"

download_one() {
    local fname="$1"
    local dest_file="$DEST/$fname"
    if [ -f "$dest_file" ]; then
        local size
        size=$(stat -f%z "$dest_file" 2>/dev/null || stat -c%s "$dest_file" 2>/dev/null || echo 0)
        if [ "$size" -gt 104857600 ]; then
            echo "[$(date)] SKIP $fname" >> "$LOG"
            return 0
        fi
    fi
    echo "[$(date)] START $fname" >> "$LOG"
    if curl -s -S --retry 3 --retry-delay 5 -C - -o "$dest_file" "$BASE_URL/$fname" 2>>"$LOG"; then
        local size
        size=$(stat -f%z "$dest_file" 2>/dev/null || stat -c%s "$dest_file" 2>/dev/null || echo 0)
        echo "[$(date)] DONE  $fname (${size} bytes)" >> "$LOG"
    else
        echo "[$(date)] FAIL  $fname" >> "$LOG"
    fi
}

export -f download_one
export BASE_URL DEST LOG

printf '%s\n' "${FILES[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'download_one "$@"' _ {}

FAILED=0
for f in "${FILES[@]}"; do
    dest="$DEST/$f"
    if [ ! -f "$dest" ]; then
        FAILED=$((FAILED+1))
    else
        sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null || echo 0)
        [ "$sz" -lt 104857600 ] && FAILED=$((FAILED+1))
    fi
done

echo "[$(date)] COMPLETE — $TOTAL total, $FAILED failed" | tee -a "$LOG"
