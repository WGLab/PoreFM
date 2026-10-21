"""Public model prefetching without the Hugging Face SDK or a tensor runtime."""

from porefm.cli._commands import Command, dispatch_commands


COMMANDS = {
    'download': Command('porefm.tools.download_model', 'Download and verify a versioned public model into the PoreFM cache.'),
}


def main(argv=None):
    """Dispatch model-management commands while keeping help lightweight."""
    return dispatch_commands(COMMANDS, argv, program_name='porefm models',
                             description='Download public PoreFM releases for reuse and offline inference.')


if __name__ == '__main__':
    raise SystemExit(main())
