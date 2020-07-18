#include <cstdio>
#include <cstring>

void train(char* argv[])
{
    std::printf("Executing training routine\n");
}

void evaluate(char* argv[])
{
    std::printf("Executing evaluation routine\n");
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::printf("Supported commands include:\ntrain\nevaluate\n");
    }

    if (strcmp(argv[1], "train") == 0)
    {
        train(argv + 2);
    }
    else if (strcmp(argv[1], "evaluate") == 0)
    {
        evaluate(argv + 2);
    }
    else
    {
        std::printf("Argument %s is an unrecognized directive.\n", argv[1]);
    }

    return 0;
}