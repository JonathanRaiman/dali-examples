def add_bool_flag(parser, name, default_value):
    parser.add_argument("--{}".format(name), action="store_true", default=default_value)
    parser.add_argument("--no{}".format(name), action="store_false", dest=name)
