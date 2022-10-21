from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
import re

class PreProcessor():
    """ A PreProcessing class so as to not instantiate multiple versions of the tokenizer in short
    succession needlessly"""
    def __init__(self):
        self.wt = WordTokenizer()
        self.st = SentenceTokenizer()
    

    def preprocess(self, inputText: str, keepPunct: bool = True) -> [str]:
        text = []
        inputText = inputText.lower()
        sents = self.st.tokenize(inputText)

        for sent in sents:
            tmpSent = []
            if not keepPunct:
                sent = re.sub(r'[^\w\s]', '', sent)
            
            wordToks = self.wt.tokenize(sent)
            input(wordToks)
            text+=wordToks
            input(text)
        return text 

if __name__=="__main__":
    exampleText = "bonam mulierem eorumque cano sunt intus dominari oportet, curam habentem omnium secundum scriptas leges, non permittentem ingredi nullum si non vir perceperit, timentem precipue verba forensium mulierum ad corrupcionem anime. et que intus sibi contingunt ut sola sciat, et si quid sinistri ab ingredientibus fiat, vir habet causam. dominam existentem expensarum et sumptuum ad festivitates quas quidem vir permiserit, expensis et vestimento ac apparatu minori utentem quam eciam leges civitatis precipiunt, considerantem quoniam nec questus vestimentorum differens forma nec auri multitudo tanta est ad mulieris virtutem quanta modestia in quolibet opere et desiderium honeste atque composite vite. etenim quilibet talis ornatus et elacio anime est, et multo cercius ad senectutem iustas laudes sibi filiisque tribuendo.talium quidem igitur ipsa se inanimet mulier composite dominari. indecens enim viro videtur scire que intus fiunt. in ceteris autem omnibus viro parere intendat, nec quicquam civilium audiens, nec aliquid de hiis que ad nupcias spectare videntur velit peragere. sed cum tempus exigit proprios filios filiasve foras tradere aut recipere, tunc autem pareat quoque viro in omnibus et simul deliberet et obediat si ille preceperit, arbitrans non ita viro esse turpe eorum que domi sunt quicquam peragere, sicut mulieri que foris sunt perquirere. sed arbitrari decet vere compositam mulierem viri mores vite sue legem inponi, a deo sibi inpositos, cum nupciis et fortuna coniunctos. quos equidem si pacienter et humiliter ferat, facile reget domum, si vero non, difficilius."
    pp = PreProcessor()
    exampleText2 = 'atque haec abuterque puerve paterne nihil'
    print(pp.preprocess(exampleText2))
