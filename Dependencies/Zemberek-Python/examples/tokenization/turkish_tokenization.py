"""
Zemberek: Turkish Tokenization Example
Java Code Example: https://bit.ly/2PsLOkj
"""

from os.path import join

from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM

if __name__ == '__main__':

    ZEMBEREK_PATH: str = join('..', '..', 'bin', 'zemberek-full.jar')

    startJVM(
        getDefaultJVMPath(),
        '-ea',
        f'-Djava.class.path={ZEMBEREK_PATH}',
        convertStrings=False
    )

    TurkishTokenizer: JClass = JClass('zemberek.tokenization.TurkishTokenizer')
    Token: JClass = JClass('zemberek.tokenization.Token')

    tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT

    inp = 'Baltimore’da büyüdüm. Çocukluğumdan beri şahsen travma ve zihinsel sağlık sorunlarıyla mücadele ettim. Tepki vermek, en büyük savunma mekanizmalarımdan biri oldu. Babam, hayatımın çoğunluğunu hapishaneye girip çıkarak geçiren bir eroin bağımlısıydı, annemi beni yalnız başına büyütmek zorunda bıraktı. Çocukların görmemesi gereken şeyleri gördüm; yetişkinlerin yaşamadığı deneyimleri yaşadım. 14 yaşımdayken akut anksiyete ve orta şiddette depresyona yakalanmıştım. Daha sonra 26 yaşında Jeneralize Anksiyete ve Panik Bozukluğu ve orta-ağır majör depresif bozukluk tanısı kondu. Teşhisimden hemen sonra, doktorum tarafından hastalığımın şiddetli olduğu ve ilaç tedavisinin tek etkili seçenek olduğu söylendi ve ilaçlar başlandı.'

    print('\nToken Iterator Example:\n')

    print(f'Input = {inp}\n')

    token_iterator = tokenizer.getTokenIterator(JString(inp))
    for token in token_iterator:
        print((
            f'Token = {token}'
            f'\n | Content = {token.content}'
            f'\n | Normalized = {token.normalized}'
            f'\n | Type = {token.type}'
            f'\n | Start = {token.start}'
            f'\n | End = {token.end}\n'
        ))

    print('Default Tokenization Example:\n')

    tokenizer: TurkishTokenizer = TurkishTokenizer.DEFAULT

    print(f'Input = {inp}')
    for i, token in enumerate(tokenizer.tokenizeToStrings(
            JString(inp)
    )):
        print(f' | Token String {i} = {token}')

    print('\nCustom Tokenization Example:\n')

    tokenizer: TurkishTokenizer = TurkishTokenizer.builder().ignoreTypes(
        Token.Type.Punctuation,
        Token.Type.NewLine,
        Token.Type.SpaceTab
    ).build()
    inp: str = 'Saat, 12:00'
    print(f'Input = {inp}')
    for i, token in enumerate(tokenizer.tokenize(JString(inp))):
        print(f' | Token {i} = {token}')

    shutdownJVM()
