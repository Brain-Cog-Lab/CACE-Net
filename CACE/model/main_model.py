import torch
from torch import nn
import torch.nn.functional as F
from .models import Only_Audio_Guided_Attention, Only_Visual_Guided_Attention, Co_Guided_Attention
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention
from braincog.model_zoo.fc_snn import SHD_SNN
import torch.nn.init as init
from copy import deepcopy

class SNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(SNNEncoder, self).__init__()
        self.d_model = d_model
        self.audio_snn = SHD_SNN(step=15, tau=2.0, threshold=0.5, layer_by_layer=True, input_dim=audio_dim, hidden_size=d_model).cuda()
        self.visual_snn = SHD_SNN(step=15, tau=2.0, threshold=0.5, layer_by_layer=True, input_dim=video_dim, hidden_size=2*d_model).cuda()

    def forward(self, audio_feature, visual_feature):
        audio_output = self.audio_snn(audio_feature)
        video_output = self.visual_snn(visual_feature)
        return audio_output, video_output


class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output

class RNNOneDirectionEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNOneDirectionEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, d_model, num_layers=num_layers, batch_first=True,
                                 bidirectional=False, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, d_model*2, num_layers=num_layers, batch_first=True, bidirectional=False,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output

class FcEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(FcEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.Linear(audio_dim, d_model)
        self.visual_rnn = nn.Linear(video_dim, d_model*2)

    def forward(self, audio_feature, visual_feature):
        audio_output = self.audio_rnn(audio_feature)
        video_output = self.visual_rnn(visual_feature)
        return audio_output, video_output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class+1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, content):

        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model, contrastive, Lambda=0.0):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.Lambda = Lambda
        self.relu = nn.ReLU(inplace=True)
        self.contrastive = contrastive

        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.encoder_q = nn.Sequential(
                         nn.Linear(d_model, d_model),
                         nn.ReLU()
        )
        self.event_classifier = nn.Linear(d_model, 28)

    def forward(self, fused_content, contrastive_switch):
        """
        :param fused_content: [10, batch, 256]
        contrastive_switch: False
        :return:
        """
        query_fused_content = self.encoder_q(fused_content)
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)

        if self.contrastive is True:
            output_content = fused_content
            if contrastive_switch is True:
                output_content = output_content + self.Lambda * query_fused_content
            logits = self.classifier(output_content)
        else:
            logits = self.classifier(fused_content)

        query_fused_content = nn.functional.normalize(query_fused_content, dim=2).transpose(1, 0)  # [N, T, C]

        key_fused_content = self.encoder_q(self.data_augmentation(fused_content))
        key_fused_content = nn.functional.normalize(key_fused_content, dim=2).transpose(1, 0)  # [N, T, C]

        class_logits = self.event_classifier(max_fused_content)

        class_scores = class_logits

        return logits, class_scores, query_fused_content, key_fused_content


    def data_augmentation(self, fused_content, num_masks=1, mask_size=1):
        # # ----------特征级噪声添加-------#
        noise = torch.randn_like(fused_content) * 0.1 # 假设噪声水平为0.1/

        # # # ----------channel mask-------#
        # # num_masks 是掩码的数量，mask_size 是每个掩码的大小
        # N, T, C = fused_content.size()
        # mask = torch.ones_like(fused_content)
        # for _ in range(num_masks):
        #     c = torch.randint(0, C - mask_size, (1,)).item()
        #     mask[:, :, c:c + mask_size] = 0

        # ----------mixup-------#
        # N, T, C = fused_content.size()
        # # 创建一个随机排列的索引，用于选择时间点进行混合
        # indices = torch.randperm(T).to(fused_content.device)
        # # 初始化一个与输入形状相同的张量，用于存储增强后的特征
        # mixed_content = torch.zeros_like(fused_content).to(fused_content.device)
        # alpha = 0.1
        # for t in range(T):
        #     # 选择用于混合的时间点，通过在时间轴上索引
        #     mix_t = indices[t]
        #     # 计算混合特征
        #     mixed_content[:, t, :] = fused_content[:, t, :] + alpha * fused_content[:, mix_t, :]

        fused_content_augmented = fused_content + noise
        # fused_content_augmented = fused_content * mask

        return fused_content_augmented

class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim, contrastive, eta=0.0):
        super(WeaklyLocalizationModule, self).__init__()
        self.eta = eta
        self.relu = nn.ReLU(inplace=True)
        self.contrastive = contrastive
        self.hidden_dim = input_dim  # need to equal d_model

        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

        self.CAS_model = CAS_Module(d_model=self.hidden_dim, num_class=28)
        self.encoder_q = nn.Sequential(
                         nn.Linear(input_dim, input_dim),
                         nn.ReLU()
        )
        self.W3 = nn.Linear(29, 1, bias=False)

    def forward(self, fused_content, contrastive_switch, video_cas_gate, audio_cas_gate):
        """
        :param fused_content: [10, batch, 256]
        contrastive_switch: False
        audio_visual_gate: [batch, 10, 1]
        :return:
        """

        query_fused_content = self.encoder_q(fused_content).transpose(0, 1)

        fused_content = fused_content.transpose(0, 1)  # [B, T, C]
        origin_fused_content = fused_content

        # max_fused_content, _ = fused_content.max(1)  # 原始的在这里
        # cas_score = self.CAS_model(fused_content)
        #
        # cas_score = 0.5 * video_cas_gate * cas_score + 0.5 * audio_cas_gate * cas_score  # gamma

        # cas_score = fused_content
        # cas_score = cas_score*2
        # sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        # topk_scores = sorted_scores[:, :4, :]
        # raw_logits = torch.mean(topk_scores, dim=1)[:, None, :]  # [32, 29]

        if self.contrastive is True:
            output_content = fused_content
            if contrastive_switch is True:
                fused_content = output_content + self.eta * query_fused_content
        else:
            fused_content = fused_content

        av_gate = self.classifier(fused_content)

        query_fused_content = nn.functional.normalize(query_fused_content, dim=2)  # [N, T, C]

        key_fused_content = self.encoder_q(self.data_augmentation(fused_content))
        key_fused_content = nn.functional.normalize(key_fused_content, dim=2)  # [N, T, C]

        # CMBS
        cas_score = self.CAS_model(origin_fused_content)
        cas_score = 0.5*video_cas_gate*cas_score + 0.5*audio_cas_gate*cas_score
        sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :4, :]
        raw_logits = torch.mean(topk_scores, dim=1)[:, None, :]       #[32, 29]

        # max_fused_content, _ = origin_fused_content.max(1)  # 对照结果在这里
        # raw_logits = self.event_classifier(max_fused_content)[:, None, :]  # [bs, 1, 29]

        fused_logits = av_gate.sigmoid() * raw_logits  # [N, T, C]

        ####################################### weighting branch #######################
        # temporal_wei = self.relu(self.W3(fused_logits)) # [bs, 10, 1]
        # temporal_wei = torch.sigmoid(temporal_wei)
        # calibration_fused_logits = fused_logits + fused_logits * temporal_wei.expand_as(fused_logits)  # [B, T, C]

        #################################################################################
        logits, _ = torch.max(fused_logits, dim=1)
        # fused_logits = calibration_fused_logits.permute(0, 2, 1)  # [B, C, T]
        #
        # out = nn.AvgPool1d(fused_logits.shape[2])(fused_logits).view(fused_logits.shape[0], -1)
        event_scores = self.softmax(logits)

        return av_gate.squeeze(), raw_logits.squeeze(), event_scores, query_fused_content, key_fused_content


    def data_augmentation(self, fused_content, num_masks=1, mask_size=1):
        # 特征级噪声添加
        noise = torch.randn_like(fused_content) * 0.1 # 假设噪声水平为0.1/
        fused_content_augmented = fused_content + noise

        return fused_content_augmented

class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class weak_main_model(nn.Module):
    def __init__(self, config, psai=0.0, guide=False, contrastive=False, eta=0.0):
        super(weak_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.psai = psai
        self.guide = guide

        self.video_input_dim = self.config['video_inputdim']
        self.audio_input_dim = self.config['audio_inputdim']

        self.video_fc_dim = 512
        self.audio_fc_dim = 128
        self.d_model = self.config['d_model']

        if self.dual is True:
            self.spatial_channel_att = Co_Guided_Attention(self.psai, self.beta, self.video_fc_dim, self.audio_fc_dim).cuda()
        else:
            self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()

        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)  # self.video_fc_dim
        self.a_fc = nn.Linear(self.audio_input_dim, self.audio_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.audio_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.audio_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        #self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=512, d_model=256, num_layers=1)
        self.AVInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.localize_module = WeaklyLocalizationModule(self.d_model, contrastive=contrastive, eta=eta)
        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=-1)
        self.audio_cas = nn.Linear(self.d_model, 29)
        self.video_cas = nn.Linear(self.d_model, 29)
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']
        self.contrastive_switch = False

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # this fc is optinal, that is used for adaption of different visual features (e.g., vgg, resnet).
        #audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        if self.dual is True:
            visual_feature, audio_feature = self.spatial_channel_att(visual_feature, audio_feature)  # output [B, T, C]
        else:
            visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        #visual_rnn_input = visual_feature


        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        # audio_gate = self.audio_gated(video_key_value_feature)
        # video_gate = self.video_gated(audio_key_value_feature)
        # audio_visual_gate = (audio_gate + video_gate) / 2
        # audio_visual_gate = audio_visual_gate.permute(1, 0, 2)
        #
        # video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        # audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha

        video_cas = self.video_cas(video_query_output)
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)

        video_cas_gate = video_cas.sigmoid()
        audio_cas_gate = audio_cas.sigmoid()
        #
        # video_cas_gate = (video_cas_gate > 0.01).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.01).float()*audio_cas_gate

        # video_cas = audio_cas_gate.unsqueeze(1) * video_cas
        # audio_cas = video_cas_gate.unsqueeze(1) * audio_cas
        #
        # sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        # topk_scores_video = sorted_scores_video[:, :4, :]
        # score_video = torch.mean(topk_scores_video, dim=1)
        # sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        # topk_scores_audio = sorted_scores_audio[:, :4, :]
        # score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 29]
        #
        # video_cas_gate = score_video.sigmoid()
        # audio_cas_gate = score_audio.sigmoid()
        # video_cas_gate = (video_cas_gate > 0.5).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.5).float()*audio_cas_gate

        #
        # av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        # --------对比学习在这里开始做---------#
        av_gate, raw_logits, event_scores, contrastive_feature1, contrastive_feature2  = self.localize_module((video_query_output+audio_query_output)/2, self.contrastive_switch, video_cas_gate, audio_cas_gate) # [T, B, C]

        return av_gate, raw_logits, event_scores, contrastive_feature1, contrastive_feature2


class supv_main_model(nn.Module):
    def __init__(self, config, psai=0.0, guide=False, contrastive=False, Lambda=0.0):
        super(supv_main_model, self).__init__()
        self.config = config
        self.beta = self.config["beta"]
        self.psai = psai
        self.guide = guide

        self.video_input_dim = self.config['video_inputdim']
        self.audio_input_dim = self.config['audio_inputdim']

        self.video_fc_dim = 512  # 512
        self.audio_fc_dim = 128  # 128
        self.d_model = self.config['d_model']

        if self.guide == "Co-Guide":
            self.spatial_channel_att = Co_Guided_Attention(self.psai, self.beta, self.video_fc_dim, self.audio_fc_dim).cuda()
        elif self.guide == "Audio-Guide":
            self.spatial_channel_att = Co_Guided_Attention(0.0, self.beta, self.video_fc_dim, self.audio_fc_dim).cuda()  # 注意这里的self.psai 需要为0.0
        elif self.guide == "Visual-Guide":
            self.spatial_channel_att = Only_Visual_Guided_Attention(self.psai).cuda()

        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)  # self.video_fc_dim
        self.a_fc = nn.Linear(self.audio_input_dim, self.audio_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=1024)  # self.video_fc_dim
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=512)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=512)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_fc_dim, video_dim=self.video_fc_dim, d_model=self.d_model, num_layers=1)  # self.audio_fc_dim
        # self.audio_visual_rnn_layer = FcEncoder(audio_dim=self.audio_fc_dim, video_dim=self.video_fc_dim, d_model=self.d_model, num_layers=1)
        # self.audio_visual_rnn_layer = RNNOneDirectionEncoder(audio_dim=self.audio_fc_dim, video_dim=self.video_fc_dim, d_model=self.d_model, num_layers=1)
        # self.audio_visual_rnn_layer = SNNEncoder(audio_dim=self.audio_fc_dim, video_dim=self.video_fc_dim, d_model=self.d_model, num_layers=1)



        self.audio_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )
        self.video_gated = nn.Sequential(
                        nn.Linear(self.d_model, 1),
                        nn.Sigmoid()
                    )

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model, contrastive=contrastive, Lambda=Lambda)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']
        self.contrastive_switch = False

    def forward(self, visual_feature, audio_feature):
    # def forward(self, inputs):
    #     visual_feature, audio_feature = inputs
        # [batch, 10, 512]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        # audio_feature = self.a_fc(audio_feature)
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        if self.guide == "None":
            audio_rnn_input = audio_feature.transpose(1, 0).contiguous()
            B, T, H, W, C = visual_feature.shape
            visual_feature = visual_feature.reshape(B, T, H*W, C).sum(dim=2) / (H * W)
        else:
            visual_feature, audio_rnn_input = self.spatial_channel_att(visual_feature, audio_feature)  # output [B, T, C]

        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)  # audio_rnn_input.shape [B, T, 128]; visual_rnn_input.shape [B, T, 512]
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]

        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)


        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha


        video_cas = self.video_cas(video_query_output)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        # event_visual_gate = score_video.sigmoid()
        # event_audio_gate = score_audio.sigmoid()

        av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        is_event_scores, event_scores, contrastive_feature1, contrastive_feature2 = self.localize_module((video_query_output + audio_query_output)/2, self.contrastive_switch)
        event_scores = event_scores + self.gamma*av_score
        #event_scores = event_scores + self.gamma * (event_visual_gate * event_audio_gate) * event_scores


        return is_event_scores, event_scores, audio_visual_gate, av_score, contrastive_feature1, contrastive_feature2

