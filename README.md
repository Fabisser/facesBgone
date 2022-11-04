# My Project


Remove internal faces from [CityJSON](http://www.cityjson.org/ "CityJSON") buildings. Two methods are present, using [CGAL](https://www.cgal.org/ "CGAL") libraries. A python script to calculate adjancancy and remove internal faces is also included.

## Documentation

Read the paper [here](https://www.tudelft.nl/en/education/programmes/masters/geomatics/msc-geomatics/programme/synthesis-project/)

## Requirements

This project uses C++ and Python. Python packages can be install by
`pip install -r requirements`

For C++, we use CGAL. To install on Linux or Windows (WSL) use
`sudo apt-get install libcgal-dev`
To install on MacOS use
`brew install cgal`

## Usage

Run the bash file.
`bash something.sh`

All the possible options can be checked using
`bash something.sh -h`

```markdown
Usage:

	bash something.sh FILENAME.json [OPTIONS]

	Will calculate the adjancacy and remove internal faces using both methods

Options:

	-m minkowski/hole-filling	Choose one of the two methods to be applied. Both will run if this flag is not included
	-h		                Show help
```

