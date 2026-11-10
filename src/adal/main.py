"""`adal`: run the service. One process, one worker: the model lives in it."""

import logging
import os

import uvicorn

from adal.api import create_app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("ADAL_LOG", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(
        create_app(), host=os.environ.get("ADAL_HOST", "127.0.0.1"), port=int(os.environ.get("ADAL_PORT", "8700"))
    )


if __name__ == "__main__":
    main()
