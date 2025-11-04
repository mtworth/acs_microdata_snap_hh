## ACS Microdata SNAP HH 

Quick script to get some basic information on SNAP households as of the latest ACS.

https://www.census.gov/data/developers/guidance/microdata-api-user-guide.html

### CPS Food Security Supplement helper

`cps_food_security_summary.py` mirrors the exploratory notebook but focuses on the
December CPS Food Security Supplement (FSS). It pulls household-level CPS
microdata and aggregates the weighted household counts by food security status
for a requested state (California by default).

```bash
python cps_food_security_summary.py --api-key YOUR_CENSUS_KEY --state 06 --year 2023
```

The script outputs a small table with the weighted household totals for each
food security status category. A matching unit test suite that exercises the
core logic lives under `tests/`.
