# Mixed-precision Randomized SVD

## time breakdown

Uncomment this line in Makefile
```make
#CXXFLAGS+=-DTIME_BREAKDOWN
```

### make figure
```
make breakdown_aggregator
mkdir tmp
cat job.log | ./breakdown_aggregator tmp
python ./scripts/time_breakdown.py --input_dir=./tmp
```

## FP16 emulation

Uncomment this line in Makefile
```make
#CXXFLAGS+=-DFP16_EMULATION
```
