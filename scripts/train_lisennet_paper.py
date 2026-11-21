"""CLI wrapper for the isolated LiSenNet paper-recipe training track."""

from speech_denoising.training.paper_lisennet import parse_args, run

if __name__ == "__main__":
    run(parse_args())
