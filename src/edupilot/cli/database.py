"""
Database CLI
============
Schema migrations, without an alembic.ini.

    edupilot-db upgrade                    apply every pending migration
    edupilot-db current                    show the applied and latest revision
    edupilot-db revision -m "add x"        new migration, diffed from db/tables.py
    edupilot-db downgrade -1               step one migration back

The app also runs `upgrade` at startup, so this is for creating migrations
and for operating on a database the app is not running against.
"""

from __future__ import annotations

import argparse
import sys

from edupilot.core.observability import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="EduPilot database migrations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("upgrade", help="apply pending migrations")
    sub.add_parser("current", help="show the applied and latest revision")
    revision = sub.add_parser("revision", help="create a migration from db/tables.py")
    revision.add_argument("-m", "--message", required=True)
    revision.add_argument(
        "--empty", action="store_true", help="write an empty migration instead of diffing"
    )
    downgrade = sub.add_parser("downgrade", help="revert migrations")
    downgrade.add_argument("target", help='a revision id, or "-1" for one step')
    args = parser.parse_args()

    from alembic import command

    from edupilot import db
    from edupilot.db.schema import alembic_config

    if args.command == "upgrade":
        db.init_db()
        print(f"database at {db.current_revision()}")
    elif args.command == "current":
        print(f"applied : {db.current_revision() or '(none)'}")
        print(f"latest  : {db.head_revision()}")
    elif args.command == "revision":
        command.revision(alembic_config(), message=args.message, autogenerate=not args.empty)
    elif args.command == "downgrade":
        command.downgrade(alembic_config(), args.target)
        print(f"database at {db.current_revision() or '(empty)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
