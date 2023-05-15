"""
    This file is to create the random shuffle of generated model texts, which are then converted 
    into the survey
"""
import random

texts = ["olim sapiens vir nomine rufo erat. hic cum equitibus colloqui venit. idem cum suis agit liberius et magna parte militum disceditur. tum vero etiam reliquis civitatibus terror et a quibusdam dissentire videbatur. tum vero ex omnibus civitatibus principes convenerunt",
         "olim sapiens vir nomine rufo erat, a gente tua sumepossedi. de gente syarchesia primus, in illa divitiis carens bona, nulla domo, et quamvis osi, tamen uxor, et illi gratus erat ; nec tamen amissa mansit",
         "olim sapiens vir nomine rufo erat sed etiam vir et virtute divina gloria, pompeius, is nimirum ille quidem, ne consultus quidem, sed tamen artis suae defensor et temporum dico memoria dignissimus, omni memoria temporum dignissimus, vobis erit iudicio",
         "olim sapiens vir nomine rufo erat sed nunc lumine da nomine dicunt, huic cognomen ; iulius nomen tuum dicunt : cui gravido cognomen fuerat, sed aurea nomen, huic cognomen de patre fuit. cornelio ordine nemo patricigravide sanguine esse voluerunt.",
         "olim sapiens vir nomine rufo erat, oallis comminuti ex oppido prohiberent, ad quam regionem consuevit, et siciliam communicio posset, quod illum ipsi quam celerrebribet, quaeque esse aut animadverteret se consulerent.",
         "olim sapiens vir nomine rufo erat,od patiarer virgo, nec avers vitiet nivea sub axe, inchisaque variarum thybri tempus hlarum texta syracorumque implevit aesone, esse mei. adspiciens ubi iuque complecta lubra licebit; iuppitis ausis lustravit in undis",
         "olim sapiens vir nomine rufo erat,ortes moribus illi operumque bonis severitatem. ita mutabunt praesentiri sibi autem populari riges acrio esse videtur. tu ad philosophisous non hoc imaginium esse pecuniam:  vetteris platonis tum commendo exstitisse nisi",
         "olim sapiens vir nomine rufo erat,otis quos illius incidit, provolvendum pecudes; at illi specubus uter corpora tepefant mavora, ut rebus eurytis subducere claudo et magno de gentes luctantur longe commortalibus arma.",
         """Olim sapiens vir nomine Rufo erat, qui multos annos in Gallia commoratus est. Ille tempus suum in studiis et litteris consumebat, sed postea ad militiam se contulit et ad Caesarem venit. Illo tempore Caesar in Italia bellum gerebat, et Rufo statim se ad eius partes adiunxit. Cum Caesar in Galliam rediit, Rufo cum eo venit et multa pro Caesare pugnavit.
				
		Sed Rufo non solum fortis miles sed etiam sapiens vir erat. Semper cum Caesar consilium capiebat, Rufo ei consilium dabat et multa utilia suggerebat. Interdum autem cum militibus suis in castris morabatur et ad eos litteras scribebat. In una epistula scripsit: \"Hoc tempore in Gallia non solum pugnamus sed etiam discimus, ut postea sapientiores simus.\"
		
		Ita Rufo non solum fortis miles sed etiam sapiens vir fuit, cuius memoria semper apud milites Caesaremque carissima fuit.""",
        """
        Olim sapiens vir nomine Rufo erat,
Ingenio doctus, consiliisque sagax,
Linguam callidam, promptamque ad verba iocosa,
Aurea cuius erat sanguinis origo.

		Fortunae motus non impavidus horret,
Nec laudis avidus, nec acri invidia
Turbatur, sed in medio haud inscius urbis
Naturae rerum structuram scrutatur.

		Mille modis vitae ludens experiensque,
Quaeque canit, natales sua carmina vates
Mutat, et haec illi vitam tam bene ducit,
Quod carmen tantum viveret immortalis.
        """,
        """
        Olim sapiens vir nomine Rufo erat, qui in re publica multos annos honorifice versatus est. Cum illo tempore res publica maxime turbulenta esset, Rufo saepe Caesarem adversatus est, ceterosque populares homines fortiter accusavit.

		Sed Rufo non solum eloquentia sed etiam sapientia pollebat, et interdum Caesaris acerbas ambitiones vituperabat. Cuius consilia multi viri boni secuti sunt, et civitas a calamitate servata est.

		Cum autem Rufo in senatum rediret, conspectus est ut fuit omnium carus et gratus. At non solum gratiam sed etiam laudem meruit, quod consilio et virtute rem publicam servavit. Et postea inter eos qui patriae bene meruerunt numeratus est, cuius memoria in saecula saeculorum vivet.
        """,
        """
        Olim sapiens vir nomine Rufo erat, quem Romae omnes digni laude viri colebant. Ingenio doctus, et artibus liberalibus imbutus, Rufo multas scientias e studio addiscere curavit.

		Sed Rufo praecipue poesis cordi erat, et carmina elegiaca componere sapientissimus existimabatur. Namque carminibus suis, amoris et virtutis laudes celebravit, et deorum divina munia mirabili modo descripsit.

		Ingenio praeditus et forma egregia, Rufo multarum feminarum adulescentiam amorem adferre poterat. Sed, ne quis eorum amore captus duceretur, Rufo semper modestissime se gessit et abstinuit ab omni lascivia.

		Inter Graecos versatus, Rufo multa ab eis didicit et artem poeticam perfecit. Quae tam perfecte scripsit, ut ea ad aeternitatem pervenirent et omnes carminibus eius semper laeti essent.
        """
        ]
random.shuffle(texts)
with open("Shuffled Ordered Texts", "w+") as f:
    f.write(" \n ========================= \n".join(texts))