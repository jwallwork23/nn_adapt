all: install

install:
	if [ -e ${VIRTUAL_ENV}/src/firedrake ]; then \
		python3 -m pip install -r requirements.txt; \
		python3 -m pip install -e .; \
    else \
		echo "You need to install Firedrake"; \
	fi

lint:
	@echo "Checking lint..."
	@flake8 --ignore=E501,E226,E402,E731,E741,F403,F405,F999,N803,N806,W503
	@echo "PASS"
