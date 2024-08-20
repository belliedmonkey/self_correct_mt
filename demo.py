from ter_lib import generate_ans, TEaR

def demo():
    # Set arguments
    lang_pair = "zh-en"
    src_lang = "Chinese"
    tgt_lang = "English"
    #model = "gpt-3.5-turbo" 
    #model = "gpt-4"
    model = "claude-3-5-sonnet-20240620"
    translate_strategy = "few-shot"
    estimate_strategy = "few-shot"
    refine_strategy = "beta"
    src_text = "孙悟空由开天辟地以来的仙石孕育而生，因带领群猴进入水帘洞而被推举成为众猴之王，号称“美猴王”。后来筏渡南赡部洲，拜得须菩提祖师为师学艺，得名孙悟空，并习得地煞数七十二变、筋斗云等高超术法。神通初成的孙悟空先闯龙宫得宝如意金箍棒，又闹地府勾去生死簿，后被天界招安，官封弼马温。得知职位低卑后气返花果山，自封齐天大圣。而后击败受命下界讨伐的托塔天王和三太子，玉皇大帝承认其齐天大圣封号，并在天庭建齐天大圣府，受命管理蟠桃园。醉酒后搅乱王母的蟠桃盛会，偷吃太上老君的金丹，逃下界去。十万天兵天将、四大天王、二十八星宿领命对其围剿，后被太上老君金刚镯暗算，为二郎真君所擒。天庭对其束手无策，遂投入太上老君的炼丹炉中，阴差阳错间炼就火眼金睛和金刚不坏之躯。后孙悟空伺机逃出并大闹天宫，逼得玉皇大帝西请如来，被如来用五指化作的五行山（唐时更名为两界山）压在山下，经历了五百多年悔过自新。后来经观音点化，被唐僧救出，法号行者，保护唐僧西天取经，一路降妖除魔，不畏艰难困苦，历经九九八十一难，最后取得真经修成正果，得封斗战胜佛。"

    # Initialize TEaR instances
    T = TEaR(lang_pair=lang_pair, model=model, module='translate', strategy=translate_strategy)
    E = TEaR(lang_pair=lang_pair, model=model, module='estimate', strategy=estimate_strategy)
    R = TEaR(lang_pair=lang_pair, model=model, module='refine', strategy=refine_strategy)

    # Load examples and set up the parser
    examples = T.load_examples() # if few-shot translate is not supported, automatically use zero-shot translate
    json_parser, json_output_instructions = T.set_parser()

    # Translate
    T_messages = T.fill_prompt(src_lang, tgt_lang, src_text, json_output_instructions, examples)
    hyp = generate_ans(model, 'translate', T_messages, json_parser)

    # Estimate
    json_parser, json_output_instructions = E.set_parser()
    E_messages = E.fill_prompt(src_lang, tgt_lang, src_text, json_output_instructions, examples, hyp)
    mqm_info, nc = generate_ans(model, 'estimate', E_messages, json_parser)

    # Refine if necessary
    if nc == 1:
        json_parser, json_output_instructions = R.set_parser()
        R_messages = R.fill_prompt(src_lang, tgt_lang, src_text, json_output_instructions, examples, hyp, mqm_info)
        cor = generate_ans(model, 'refine', R_messages, json_parser)
    elif nc == 0:
        cor = hyp

    # Display translation results
    print(f"----------------(╹ڡ╹ )----------TEaR---------o(*￣▽￣*)ブ-----------------")
    print(f"Model: {model}")
    print(f"Source: {src_text}")
    print(f"Hypothesis: {hyp}")
    print(f"Correction: {cor}")
    print(f"Need correction: {nc}")
    print(f"MQM Info: {mqm_info}")


if __name__ == '__main__':
    demo()
