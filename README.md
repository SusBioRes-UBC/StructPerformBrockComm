## This repo contains the time series analysis models for operational performance of CLT in UBC Brock Commons Tallwood House

### [Link to the report](https://sustain.ubc.ca/sites/default/files/UBC%20Brock%20Commons%20Structural%20Performance%20Report%20Sept%202020.pdf)
### [Link to climate data](https://www.mkrf.forestry.ubc.ca/research/weather-data/?login) 


## Dependencies
### Prophet (which depends on pystan): https://facebook.github.io/prophet/docs/installation.html
- [caution] you need to install a compiler before installing **pystan** on Windows OS --> install "Desktop development with C++" via "Microsoft Studo Build Tools". Alternatively, you can install pystan via Anaconda, [see solution](https://medium.com/@hamdanmridwan/quickly-setting-up-prophet-with-python-3-x-in-windows-10-ad92aaaa081d)
- [caution] error when install *pip install prophet* after pystan is installed, [see solutin](https://hemantjain.medium.com/solution-for-the-error-while-installing-prophet-library-on-windows-machine-d1cc84adbafc) 